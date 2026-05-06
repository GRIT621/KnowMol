from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class ProteinAgent(BaseAgent):
    """Generate protein target fragments used by KnowMol sequence-fragment features."""

    def __init__(
        self,
        name: str,
        dataset: Any,
        model: str = "pro-deepseek-r1",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(name, model, api_key, api_base)
        self.dataset = dataset

    def _build_prompt(self, samples: list[dict[str, Any]]) -> str:
        return f"""
You are a drug target expert. The following is a sample of drug-target records:
{samples}

Please identify and list ten distinct protein target sub-fragments that are likely to influence binding affinity or interaction labels.

Requirements:
- Each target sub-fragment must be strict, unique, and independently predictive.
- There is no fixed fragment length, but do not return full-length protein sequences.
- Return protein fragments as a numbered list only.
- No explanations.

Example:
1. AYIHSFGICHRDIK
2. AAASTPTNATAASDANTGDRGQTNNAA
""".strip()

    def generate_substructure(self, samples: list[dict[str, Any]]) -> str:
        prompt = self._build_prompt(samples)
        response = self._call_model([{"role": "user", "content": prompt}])
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if choice.finish_reason == "length":
            raise RuntimeError("ProteinAgent response was truncated; retry with a larger model limit.")
        if not content:
            raise RuntimeError(f"ProteinAgent returned empty content: finish_reason={choice.finish_reason}")
        return content.strip()
