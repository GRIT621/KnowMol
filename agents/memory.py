from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ShortMemory:
    """Per-round memory: sampled records, agent outputs, validation feedback."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    @classmethod
    def load(cls, path: Path) -> "ShortMemory":
        memory = cls()
        if not path.exists():
            return memory
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return memory
        if isinstance(data, list):
            memory.records = [record for record in data if isinstance(record, dict)]
        return memory

    def start_round(self, round_index: int, samples: list[dict[str, Any]]) -> None:
        self.records.append(
            {
                "round": round_index + 1,
                "samples": samples,
                "drug_agent_output": "",
                "target_agent_output": "",
                "candidate_drug_features": [],
                "candidate_target_features": [],
                "metrics": {},
                "validation_gain": None,
                "consolidated": False,
                "badcases": [],
            }
        )

    def set_agent_output(self, key: str, text: str) -> None:
        if not self.records:
            return
        self.records[-1][key] = text

    def set_candidate_features(self, drug_features: list[str], target_features: list[str]) -> None:
        if not self.records:
            return
        self.records[-1]["candidate_drug_features"] = drug_features
        self.records[-1]["candidate_target_features"] = target_features

    def set_validation_feedback(self, metrics: dict[str, float], badcases: pd.DataFrame, limit: int = 10) -> None:
        if not self.records:
            return
        self.records[-1]["metrics"] = metrics
        columns = [column for column in ["drug", "target", "label", "_feedback_error"] if column in badcases.columns]
        self.records[-1]["badcases"] = badcases.head(limit)[columns].to_dict("records")

    def set_consolidation(self, validation_gain: float, consolidated: bool) -> None:
        if not self.records:
            return
        self.records[-1]["validation_gain"] = validation_gain
        self.records[-1]["consolidated"] = consolidated

    def write_round(self, path: Path) -> None:
        if not self.records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records[-1], indent=2, ensure_ascii=False))

    def write_all(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False))

    def build_agent_context(
        self,
        drug_memory: dict[str, Any],
        target_memory: dict[str, Any],
        badcases: pd.DataFrame | None = None,
        feature_limit: int = 12,
        badcase_limit: int = 5,
        recent_round_limit: int = 3,
    ) -> str:
        """Compact memory summary for the next Drug/Target Agent prompts."""
        lines: list[str] = []
        lines.extend(self._format_feature_memory("Long-memory drug SMARTS", drug_memory, feature_limit))
        lines.extend(self._format_feature_memory("Long-memory target fragments", target_memory, feature_limit))

        rejected_drug: list[str] = []
        rejected_target: list[str] = []
        for record in reversed(self.records[-recent_round_limit:]):
            if record.get("consolidated"):
                continue
            rejected_drug.extend(record.get("candidate_drug_features", []))
            rejected_target.extend(record.get("candidate_target_features", []))
        rejected_drug = list(dict.fromkeys(rejected_drug))[:feature_limit]
        rejected_target = list(dict.fromkeys(rejected_target))[:feature_limit]
        if rejected_drug:
            lines.append("Recently rejected drug candidates: " + ", ".join(rejected_drug))
        if rejected_target:
            lines.append("Recently rejected target candidates: " + ", ".join(rejected_target))

        feedback = self._format_badcases(badcases, badcase_limit)
        if feedback:
            lines.append("Current validation badcases:")
            lines.extend(feedback)

        if not lines:
            return "No previous memory is available yet."
        return "\n".join(lines)

    def _format_feature_memory(self, title: str, features: dict[str, Any], limit: int) -> list[str]:
        if not features:
            return [f"{title}: none"]
        ranked = sorted(features.items(), key=lambda item: self._memory_score(item[1]), reverse=True)
        formatted = []
        for key, value in ranked[:limit]:
            entry = value if isinstance(value, dict) else {}
            gamma = entry.get("gamma", {}) if isinstance(entry.get("gamma", {}), dict) else {}
            alpha = float(entry.get("alpha", 0.0) or 0.0)
            cumulative_gain = float(gamma.get("cumulative_validation_gain", 0.0) or 0.0)
            recurrence = int(gamma.get("recurrence", 0) or 0)
            count = int(entry.get("count", 0) or 0)
            formatted.append(
                f"{self._truncate(str(key), 80)} "
                f"(alpha={alpha:.4g}, gain={cumulative_gain:.4g}, recurrence={recurrence}, count={count})"
            )
        return [f"{title}:", *[f"- {item}" for item in formatted]]

    def _memory_score(self, value: Any) -> tuple[float, float, int]:
        if not isinstance(value, dict):
            return (0.0, 0.0, 0)
        gamma = value.get("gamma", {})
        if not isinstance(gamma, dict):
            gamma = {}
        return (
            float(gamma.get("cumulative_validation_gain", 0.0) or 0.0),
            float(value.get("alpha", 0.0) or 0.0),
            int(value.get("count", 0) or 0),
        )

    def _format_badcases(self, badcases: pd.DataFrame | None, limit: int) -> list[str]:
        if badcases is None or badcases.empty:
            return []
        columns = [column for column in ["drug", "target", "label", "_feedback_error"] if column in badcases.columns]
        lines = []
        for idx, row in enumerate(badcases.head(limit)[columns].to_dict("records"), start=1):
            drug = self._truncate(str(row.get("drug", "")), 120)
            target = self._truncate(str(row.get("target", "")), 120)
            label = row.get("label", "")
            error = row.get("_feedback_error", "")
            error_text = f"{float(error):.4f}" if isinstance(error, (int, float)) else str(error)
            lines.append(f"- {idx}. label={label} error={error_text} drug={drug} target={target}")
        return lines

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
