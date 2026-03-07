import os
import json
import time
import requests
from typing import Dict, Any, Optional
from requests.exceptions import RequestException

class LLMClient:
    """
    A unified client for making API calls to LLM providers.
    Supports OpenAI API format (works with OpenAI, vLLM, Ollama, etc).
    """
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider.lower()
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
            self.base_url = base_url or "http://localhost:11434/v1"
            self.api_key = self.api_key or "ollama" # Dummy key
        else:
            self.base_url = base_url
            if not self.base_url:
                raise ValueError(
                    f"Unsupported provider '{provider}'. "
                    "Use one of: openai, anthropic, ollama, or provide a base_url."
                )
            
    def _make_request(self, messages: list, model: str, temperature: float = 0.7, 
                      max_tokens: int = 1000, response_format: Optional[dict] = None) -> str:
        max_retries = 3
        backoff = 2

        if self.provider == "anthropic":
            system_prompt = "\n".join(
                m.get("content", "") for m in messages if m.get("role") == "system"
            ).strip()
            anthropic_messages = []
            for msg in messages:
                role = msg.get("role")
                if role not in ("user", "assistant"):
                    continue
                anthropic_messages.append(
                    {"role": role, "content": msg.get("content", "")}
                )

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
                            time.sleep(backoff ** attempt)
                            continue
                    raise Exception(
                        f"LLM API Error: {str(e)} - {getattr(e.response, 'text', '')}"
                    )
            raise Exception("Max retries exceeded")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if response_format:
            # Some providers like Ollama might not support response_format strictly,
            # but standard OpenAI spec does.
            payload["response_format"] = response_format

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code == 429 or (status_code and status_code >= 500):
                    if attempt < max_retries - 1:
                        time.sleep(backoff ** attempt)
                        continue
                raise Exception(f"LLM API Error: {str(e)} - {getattr(e.response, 'text', '')}")

        raise Exception("Max retries exceeded")

    def generate_text(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
        """Generate plain text response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self._make_request(messages, model, temperature)

    def generate_json(self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> Dict[str, Any]:
        """Generate and parse a JSON response."""
        messages = [
            {"role": "system", "content": system_prompt + "\n\nReturn EXACTLY valid JSON and nothing else. No markdown blocks."},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use JSON mode if supported
        response_format = {"type": "json_object"}
        
        raw_response = self._make_request(messages, model, temperature, response_format=response_format)
        
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
            raise ValueError(f"Failed to parse LLM response as JSON. Response: {raw_response[:100]}... Error: {str(e)}")
