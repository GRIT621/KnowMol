from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class AnalysisAgent:
    """Create lightweight multi-level analysis artifacts from discovery rounds."""

    LEVELS = [
        "Global Interaction Level",
        "Target-Specific Ligand Level",
        "Ligand-Specific Protein Level",
        "Pairwise Binding Level",
    ]

    def write_round_report(
        self,
        path: str | Path,
        dataset: str,
        round_records: list[dict[str, Any]],
        substructure_count: int,
        fragment_count: int,
    ) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(round_records)
        best_auc = summary["auc_roc"].max() if "auc_roc" in summary and not summary.empty else None

        lines = [
            "# KnowMol Multi-Level Analysis",
            "",
            f"- Dataset: `{dataset}`",
            f"- Drug feature count: `{substructure_count}`",
            f"- Target feature count: `{fragment_count}`",
            f"- Best AUC-ROC: `{best_auc:.4f}`" if best_auc is not None else "- Best AUC-ROC: `N/A`",
            "",
            "## Levels",
            "",
        ]
        lines.extend(f"- {level}" for level in self.LEVELS)
        lines.extend(["", "## Round Metrics", ""])
        if summary.empty:
            lines.append("No discovery rounds were run.")
        else:
            lines.append("```text")
            lines.append(summary.to_string(index=False))
            lines.append("```")
        output.write_text("\n".join(lines))
