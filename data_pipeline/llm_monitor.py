"""
LLM Monitor Module
==================
Tracks model performance, compares runs, and triggers retraining
for continuous LLM evaluation and improvement.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class LLMMonitor:
    """
    Monitors LLM fine-tuning runs, logs evaluation metrics, compares
    model versions, and provides retraining recommendations.

    Metrics tracked:
    - Training loss
    - Perplexity
    - BLEU score
    - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    - Custom metrics

    Example:
    --------
    >>> monitor = LLMMonitor("./model_logs")
    >>> monitor.log_run("v1.0", {
    ...     "loss": 0.45, "perplexity": 12.3,
    ...     "bleu": 0.32, "rouge_1": 0.48
    ... })
    >>> monitor.compare_runs("v1.0", "v1.1")
    >>> monitor.check_retraining_triggers()
    """

    STANDARD_METRICS = [
        "loss",
        "perplexity",
        "bleu",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "accuracy",
        "f1",
    ]

    # Default retraining thresholds
    DEFAULT_TRIGGERS = {
        "perplexity_max": 50.0,
        "loss_max": 2.0,
        "bleu_min": 0.1,
        "rouge_1_min": 0.2,
        "accuracy_min": 0.5,
    }

    def __init__(
        self, log_dir: str = "./llm_logs", triggers: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the monitor.

        Parameters
        ----------
        log_dir : str
            Directory to store evaluation logs.
        triggers : dict, optional
            Custom retraining trigger thresholds.
            Keys: metric_name + _max or _min suffix.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.triggers = triggers or self.DEFAULT_TRIGGERS
        self._runs = self._load_runs()

    def _load_runs(self) -> Dict[str, Any]:
        """Load existing run logs."""
        runs_file = self.log_dir / "runs.json"
        if runs_file.exists():
            with open(runs_file, "r") as f:
                return json.load(f)
        return {"runs": {}, "latest": None}

    def _save_runs(self):
        """Persist run logs to disk."""
        runs_file = self.log_dir / "runs.json"
        with open(runs_file, "w") as f:
            json.dump(self._runs, f, indent=2, default=str)

    def log_run(
        self,
        run_id: str,
        metrics: Dict[str, float],
        model_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        notes: str = "",
        training_config: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Log an evaluation run with metrics.

        Parameters
        ----------
        run_id : str
            Unique run identifier (e.g., "v1.0", "exp-2024-03-01").
        metrics : dict
            Metric name → value mapping.
        model_name : str, optional
            Name/path of the model evaluated.
        dataset_version : str, optional
            Dataset version used for training/evaluation.
        notes : str
            Free-form notes about this run.
        training_config : dict, optional
            Training configuration used.

        Returns
        -------
        dict
            The logged run entry.
        """
        run_entry = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_name": model_name,
            "dataset_version": dataset_version,
            "notes": notes,
            "training_config": training_config,
            "dataset_stats": kwargs.get("dataset_stats", {}),
        }

        self._runs["runs"][run_id] = run_entry
        self._runs["latest"] = run_id

        # Save individual run file too
        run_file = self.log_dir / f"{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run_entry, f, indent=2)

        self._save_runs()

        print(f"📊 Logged run: {run_id}")
        for name, value in metrics.items():
            print(f"   • {name}: {value:.4f}")

        return run_entry

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get a specific run's data."""
        if run_id not in self._runs["runs"]:
            raise ValueError(
                f"Run '{run_id}' not found. "
                f"Available: {list(self._runs['runs'].keys())}"
            )
        return self._runs["runs"][run_id]

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all logged runs sorted by timestamp."""
        runs = list(self._runs["runs"].values())
        runs.sort(key=lambda r: r.get("timestamp", ""))
        return runs

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        """
        Compare metrics between two runs.

        Parameters
        ----------
        run_id_a : str
            First (baseline) run ID.
        run_id_b : str
            Second (comparison) run ID.

        Returns
        -------
        dict
            Comparison report with deltas and improvements.
        """
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        metrics_a = run_a["metrics"]
        metrics_b = run_b["metrics"]

        all_metrics = set(list(metrics_a.keys()) + list(metrics_b.keys()))

        comparisons = {}
        improvements = []
        regressions = []

        for metric in sorted(all_metrics):
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)

            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                pct_change = (delta / abs(val_a) * 100) if val_a != 0 else 0

                # Determine if improvement or regression
                # Lower is better for: loss, perplexity
                # Higher is better for: bleu, rouge, accuracy, f1
                lower_better = metric in ("loss", "perplexity")
                is_improved = (delta < 0) if lower_better else (delta > 0)

                comp = {
                    "run_a": val_a,
                    "run_b": val_b,
                    "delta": delta,
                    "pct_change": round(pct_change, 2),
                    "improved": is_improved,
                }
                comparisons[metric] = comp

                if is_improved:
                    improvements.append(metric)
                elif abs(pct_change) > 1:  # Ignore negligible changes
                    regressions.append(metric)
            else:
                comparisons[metric] = {
                    "run_a": val_a,
                    "run_b": val_b,
                    "delta": None,
                    "pct_change": None,
                    "improved": None,
                }

        report = {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "comparisons": comparisons,
            "improvements": improvements,
            "regressions": regressions,
            "overall_improved": len(improvements) > len(regressions),
        }

        return report

    def check_retraining_triggers(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if a run's metrics trigger a retraining recommendation.

        Parameters
        ----------
        run_id : str, optional
            Run to check. Defaults to latest.

        Returns
        -------
        dict
            Trigger evaluation with recommendations.
        """
        if run_id is None:
            run_id = self._runs.get("latest")
        if not run_id:
            return {"should_retrain": False, "reason": "No runs logged yet"}

        run = self.get_run(run_id)
        metrics = run["metrics"]

        triggered = []
        passed = []

        for trigger_key, threshold in self.triggers.items():
            # Parse trigger: metric_name + _max or _min
            parts = trigger_key.rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in ("max", "min"):
                continue

            metric_name = parts[0]
            direction = parts[1]

            if metric_name not in metrics:
                continue

            value = metrics[metric_name]

            if direction == "max" and value > threshold:
                triggered.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "direction": f"{metric_name} > {threshold}",
                    }
                )
            elif direction == "min" and value < threshold:
                triggered.append(
                    {
                        "metric": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "direction": f"{metric_name} < {threshold}",
                    }
                )
            else:
                passed.append(metric_name)

        should_retrain = len(triggered) > 0

        result = {
            "run_id": run_id,
            "should_retrain": should_retrain,
            "triggered": triggered,
            "passed": passed,
            "recommendation": (
                f"Retraining recommended: {len(triggered)} trigger(s) fired"
                if should_retrain
                else "Model metrics within acceptable thresholds"
            ),
        }

        return result

    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get the history of a specific metric across all runs.

        Parameters
        ----------
        metric_name : str
            Name of the metric to track.

        Returns
        -------
        list of dict
            History entries sorted by timestamp.
        """
        history = []
        for run in self.list_runs():
            value = run["metrics"].get(metric_name)
            if value is not None:
                history.append(
                    {
                        "run_id": run["run_id"],
                        "timestamp": run["timestamp"],
                        "value": value,
                        "model_name": run.get("model_name"),
                        "dataset_version": run.get("dataset_version"),
                    }
                )

        return history

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a markdown evaluation report.

        Parameters
        ----------
        output_path : str, optional
            Path to save the report. If None, returns the string.

        Returns
        -------
        str
            Markdown report content.
        """
        runs = self.list_runs()

        lines = [
            "# LLM Evaluation Report",
            f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
            "\n## Summary",
            f"\n- **Total runs**: {len(runs)}",
            f"- **Latest run**: {self._runs.get('latest', 'N/A')}",
        ]

        if runs:
            # Metrics table
            all_metrics = set()
            for run in runs:
                all_metrics.update(run["metrics"].keys())
            all_metrics = sorted(all_metrics)

            lines.append("\n## Metrics Comparison\n")
            header = "| Run | " + " | ".join(all_metrics) + " |"
            separator = "|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|"
            lines.append(header)
            lines.append(separator)

            for run in runs:
                values = []
                for m in all_metrics:
                    v = run["metrics"].get(m)
                    values.append(f"{v:.4f}" if v is not None else "—")
                line = f"| {run['run_id']} | " + " | ".join(values) + " |"
                lines.append(line)

            # Best scores
            lines.append("\n## Best Scores\n")
            for m in all_metrics:
                values = [
                    (r["run_id"], r["metrics"][m]) for r in runs if m in r["metrics"]
                ]
                if values:
                    lower_better = m in ("loss", "perplexity")
                    best = (
                        min(values, key=lambda x: x[1])
                        if lower_better
                        else max(values, key=lambda x: x[1])
                    )
                    lines.append(f"- **{m}**: {best[1]:.4f} (run: {best[0]})")

            # Retraining check
            if self._runs.get("latest"):
                trigger_result = self.check_retraining_triggers()
                lines.append("\n## Retraining Status\n")
                if trigger_result["should_retrain"]:
                    lines.append("⚠️ **Retraining recommended**\n")
                    for t in trigger_result["triggered"]:
                        lines.append(
                            f"- {t['metric']}: {t['value']:.4f} "
                            f"(threshold: {t['direction']})"
                        )
                else:
                    lines.append("✅ All metrics within acceptable thresholds.")

        report = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"📄 Report saved to: {output_path}")

        return report

    def print_summary(self) -> None:
        """Print a formatted monitoring summary."""
        runs = self.list_runs()
        print("=" * 60)
        print("LLM MONITORING SUMMARY")
        print("=" * 60)
        print(f"\n📂 Log directory: {self.log_dir}")
        print(f"📊 Total runs: {len(runs)}")
        print(f"🏷️  Latest: {self._runs.get('latest', 'none')}")

        if runs:
            latest = runs[-1]
            print(f"\n📈 Latest run ({latest['run_id']}):")
            for name, value in latest["metrics"].items():
                print(f"   • {name}: {value:.4f}")

            # Check triggers
            trigger_result = self.check_retraining_triggers()
            if trigger_result["should_retrain"]:
                print(f"\n⚠️  {trigger_result['recommendation']}")
            else:
                print(f"\n✅ {trigger_result['recommendation']}")

        print("=" * 60)
