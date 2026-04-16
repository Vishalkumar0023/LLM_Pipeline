import os
import json
import re
import time
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException


class LLMClient:
    """
    A unified client for making API calls to LLM providers.
    Supports OpenAI API format (works with OpenAI, vLLM, Ollama, etc).

    Default provider/model can be set via environment variables:
        LLM_PROVIDER   – ollama | openai | anthropic  (default: ollama)
        OLLAMA_MODEL   – model name for Ollama        (default: deepseek-r1:8b)
        OLLAMA_BASE_URL – Ollama endpoint              (default: http://localhost:11434/v1)
    """

    # DeepSeek-R1 wraps its reasoning in <think>…</think> blocks.
    # We strip them so downstream consumers get only the final answer.
    _THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if self.provider == "openai":
            self.base_url = base_url or "https://api.openai.com/v1"
            if not self.api_key:
                print("Warning: OPENAI_API_KEY not set. API calls will fail.")
        elif self.provider == "anthropic":
            self.base_url = base_url or "https://api.anthropic.com/v1"
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not self.api_key:
                print("Warning: ANTHROPIC_API_KEY not set. API calls will fail.")
        elif self.provider == "ollama":
            self.base_url = base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            self.api_key = self.api_key or "ollama"  # Dummy key for OpenAI compat
        else:
            self.base_url = base_url
            if not self.base_url:
                raise ValueError(
                    f"Unsupported provider '{provider}'. "
                    "Use one of: openai, anthropic, ollama, or provide a base_url."
                )

    # ── Ollama health probe ──────────────────────────────────────────────

    @staticmethod
    def is_ollama_available(base_url: Optional[str] = None) -> bool:
        """Return True if the Ollama server is reachable."""
        url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/v1")
        try:
            resp = requests.get(f"{url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_ollama_models(base_url: Optional[str] = None) -> "list":
        """Return a list of model names available in Ollama."""
        url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/v1")
        try:
            resp = requests.get(f"{url}/api/tags", timeout=3)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Core request ─────────────────────────────────────────────────────

    def _make_request(
        self,
        messages: list,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Optional[dict] = None,
    ) -> str:
        max_retries = 3
        backoff = 2

        # Anthropic uses a different API shape
        if self.provider == "anthropic":
            return self._make_anthropic_request(
                messages, model, temperature, max_tokens
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        # Ollama / local models need more time for large context
        timeout = 120 if self.provider == "ollama" else 60

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                raw_content = data["choices"][0]["message"]["content"]
                return self._strip_thinking(raw_content)

            except RequestException as e:
                status_code = getattr(e.response, "status_code", None)
                if status_code == 429 or (status_code and status_code >= 500):
                    if attempt < max_retries - 1:
                        time.sleep(backoff**attempt)
                        continue
                raise Exception(
                    f"LLM API Error: {str(e)} - {getattr(e.response, 'text', '')}"
                )

        raise Exception("Max retries exceeded")

    def _make_anthropic_request(
        self,
        messages: list,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        max_retries = 3
        backoff = 2

        system_prompt = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "system"
        ).strip()
        anthropic_messages = [
            {"role": msg["role"], "content": msg.get("content", "")}
            for msg in messages
            if msg.get("role") in ("user", "assistant")
        ]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("content", [])
                if content and isinstance(content, list):
                    first = content[0]
                    if isinstance(first, dict) and first.get("text"):
                        return first["text"]
                raise Exception(f"Unexpected Anthropic response schema: {data}")
            except RequestException as e:
                status_code = getattr(e.response, "status_code", None)
                if status_code == 429 or (status_code and status_code >= 500):
                    if attempt < max_retries - 1:
                        time.sleep(backoff**attempt)
                        continue
                raise Exception(
                    f"LLM API Error: {str(e)} - {getattr(e.response, 'text', '')}"
                )
        raise Exception("Max retries exceeded")

    # ── DeepSeek-R1 thinking‐token cleanup ───────────────────────────────

    @classmethod
    def _strip_thinking(cls, text: str) -> str:
        """Remove <think>…</think> blocks that DeepSeek-R1 emits."""
        cleaned = cls._THINK_PATTERN.sub("", text).strip()
        return cleaned if cleaned else text.strip()

    # ── Prompt injection sanitization ────────────────────────────────────

    # SECURITY: Known prompt injection patterns to strip from document input
    _INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*(Instruction|Response|System)",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
    ]

    @staticmethod
    def _sanitize_prompt_input(text: str, max_chars: int = 10000) -> str:
        """Strip known prompt injection patterns from document text."""
        for pattern in LLMClient._INJECTION_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text[:max_chars]

    # ── High-level generate methods ──────────────────────────────────────

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate plain text response."""
        if model is None:
            model = self._default_model()
        # SECURITY: Sanitize user prompt to prevent prompt injection
        sanitized_prompt = self._sanitize_prompt_input(user_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sanitized_prompt},
        ]
        return self._make_request(messages, model, temperature)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate and parse a JSON response."""
        if model is None:
            model = self._default_model()
        # SECURITY: Sanitize user prompt to prevent prompt injection
        sanitized_prompt = self._sanitize_prompt_input(user_prompt)
        messages = [
            {
                "role": "system",
                "content": system_prompt
                + "\n\nReturn EXACTLY valid JSON and nothing else. No markdown blocks.",
            },
            {"role": "user", "content": sanitized_prompt},
        ]

        # Use JSON mode if supported
        response_format = {"type": "json_object"}

        raw_response = self._make_request(
            messages, model, temperature, response_format=response_format
        )

        # Cleanup markdown formatting if the model still wrapped it
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        try:
            return json.loads(clean_response.strip())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON. "
                f"Response: {raw_response[:100]}... Error: {str(e)}"
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _default_model(self) -> str:
        """Return the default model name for the current provider."""
        if self.provider == "ollama":
            return os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
        if self.provider == "anthropic":
            return "claude-3-haiku-20240307"
        return "gpt-4o-mini"
