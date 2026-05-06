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

    def _format_samples(self, samples: list[dict[str, Any]], feedback_strategy: str) -> str:
        formatted = []
        for idx, sample in enumerate(samples, start=1):
            sample_copy = dict(sample)
            error_value = sample_copy.pop("_feedback_error", None)
            prefix = f"{idx}. "
            if feedback_strategy == "badcase" and error_value is not None:
                prefix += f"[badcase error={error_value:.4f}] "
            formatted.append(prefix + str(sample_copy))
        return "\n".join(formatted)

    def _build_prompt(
        self,
        samples: list[dict[str, Any]],
        excluded_fragments: Any = None,
        feedback_strategy: str = "random",
        memory_context: str = "",
    ) -> str:
        sample_block = self._format_samples(samples, feedback_strategy)
        return f"""
You are a drug target expert. The following is a sample of drug-target records:
{sample_block}

Memory context from previous validation rounds:
{memory_context or "No previous memory is available yet."}

Please identify and list ten distinct protein target sub-fragments that are likely to influence binding affinity or interaction labels.

Requirements:
- Each target sub-fragment must be strict, unique, and independently predictive.
- There is no fixed fragment length, but do not return full-length protein sequences.
- Each fragment must be new and must not repeat anything in this exclusion list:
{excluded_fragments}
- Use long-memory fragments as evidence for motifs that previously helped.
- Avoid recently rejected candidates unless a more specific fragment can address the badcase feedback.
- Return protein fragments as a numbered list only.
- No explanations.

Example:
1. AYIHSFGICHRDIK
2. AAASTPTNATAASDANTGDRGQTNNAA
""".strip()

    def generate_substructure(
        self,
        samples: list[dict[str, Any]],
        excluded_fragments: Any = None,
        feedback_strategy: str = "random",
        memory_context: str = "",
    ) -> str:
        prompt = self._build_prompt(
            samples,
            excluded_fragments,
            feedback_strategy=feedback_strategy,
            memory_context=memory_context,
        )
        response = self._call_model([{"role": "user", "content": prompt}])
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if choice.finish_reason == "length":
            raise RuntimeError("ProteinAgent response was truncated; retry with a larger model limit.")
        if not content:
            raise RuntimeError(f"ProteinAgent returned empty content: finish_reason={choice.finish_reason}")
        return content.strip()
