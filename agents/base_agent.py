from __future__ import annotations

from typing import Any


class BaseAgent:
    """Shared OpenAI-compatible chat client wrapper for KnowMol agents."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int = 800,
    ) -> None:
        self.name = name
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
        self.timeout = timeout
        self.reset_usage_stats()

    def reset_usage_stats(self) -> None:
        self.usage_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _build_prompt(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    def _call_model(self, messages: list[dict[str, str]]) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("KnowMol agents require openai>=1.0.0. Please upgrade the openai package.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model_name,
            timeout=self.timeout,
            messages=messages,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage_stats["calls"] += 1
            self.usage_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self.usage_stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        return response
