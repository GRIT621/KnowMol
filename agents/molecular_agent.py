from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class MolecularAgent(BaseAgent):
    """Generate molecule substructure SMARTS used by KnowMol fragment features."""

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
        excluded_substructures: Any,
        feedback_strategy: str = "random",
        memory_context: str = "",
    ) -> str:
        sample_block = self._format_samples(samples, feedback_strategy)
        dataset_name = getattr(self.dataset, "dataset_name", "").lower()
        task_name = {
            "bace": "BACE inhibitor activity classification",
            "bbbp": "Blood-Brain Barrier Penetration (BBBP)",
            "hiv": "HIV replication inhibition classification",
            "clintox": "clinical toxicity classification",
            "sider": "drug side-effect classification",
            "tox21": "Tox21 toxicity classification",
            "toxcast": "ToxCast assay activity classification",
            "muv": "Maximum Unbiased Validation virtual-screening classification",
            "freesolv": "aqueous solvation free energy",
            "esol": "aqueous solubility prediction",
            "lipo": "lipophilicity prediction",
        }.get(dataset_name, "Drug-Target Interaction")

        badcase_guidance = ""
        if feedback_strategy == "badcase":
            badcase_guidance = """
Important feedback context:
- These samples are the current model's highest-error bad cases.
- Prioritize substructures that can explain why these molecules are being predicted poorly.
- Focus on rare, discriminative motifs that are missing from the current substructure pool.
- Do not just return common high-frequency fragments unless they specifically help separate these failure cases.
"""

        return f"""
You are a molecular expert and RDKit SMARTS engineer.

Below is a sample of molecules from the {task_name} task:
{sample_block}

Memory context from previous validation rounds:
{memory_context or "No previous memory is available yet."}

Your task:
Identify and list ten (10) distinct and independently predictive molecular substructures that are likely to influence {task_name}.

Requirements:
- Each substructure must be chemically meaningful, unique, and non-trivial.
- Each must be represented as a valid SMARTS string compatible with RDKit's Chem.MolFromSmarts().
- Each SMARTS must represent a new substructure NOT present in the following exclusion list:
{excluded_substructures}
- Use long-memory features as evidence for what has already helped.
- Avoid recently rejected candidates unless a more specific SMARTS can address the badcase feedback.
{badcase_guidance}

Format:
Return the 10 SMARTS substructures only, numbered 1-10. No explanations or additional text.
Example:
1. *N=C(*)N
2. *S*
3. *N(*)*

Strict rule:
You must NOT repeat or partially include any of the excluded substructures. Even similar or nested patterns must be avoided.
""".strip()

    def generate_paginated(self, messages: list[dict[str, str]], max_pages: int = 5) -> str:
        full_text = ""
        page = 0

        while page < max_pages:
            response = self._call_model(messages)
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)
            reasoning_content = getattr(message, "reasoning_content", None)

            if choice.finish_reason != "length":
                final_text = (content or "").strip()
                if final_text:
                    return final_text
                raise RuntimeError(
                    f"Model returned empty content: finish_reason={choice.finish_reason}, "
                    f"usage={getattr(response, 'usage', None)}"
                )

            continuation = content or reasoning_content or ""
            full_text += continuation
            messages.append({"role": "assistant", "content": continuation})
            messages.append({"role": "user", "content": "Please continue."})
            page += 1

        final_text = full_text.strip()
        if final_text:
            return final_text
        raise RuntimeError("Model pagination returned empty content.")

    def generate_substructure(
        self,
        samples: list[dict[str, Any]],
        excluded_substructures: Any,
        feedback_strategy: str = "random",
        memory_context: str = "",
    ) -> str:
        prompt = self._build_prompt(
            samples,
            excluded_substructures,
            feedback_strategy=feedback_strategy,
            memory_context=memory_context,
        )
        return self.generate_paginated(messages=[{"role": "user", "content": prompt}])
