from __future__ import annotations

import random
import time
from types import SimpleNamespace
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
        max_retries: int = 5,
    ) -> None:
        self.name = name
        self.model_name = model
        self.api_key = api_key
        self.api_base = api_base or "https://api.openai.com/v1"
        self.timeout = timeout
        self.max_retries = max_retries
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
        response = self._call_model_with_retries(messages)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage_stats["calls"] += 1
            self.usage_stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage_stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            self.usage_stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        return response

    def _call_model_with_retries(self, messages: list[dict[str, str]]) -> Any:
        last_error: Exception | None = None
        attempts = max(1, self.max_retries + 1)

        for attempt in range(1, attempts + 1):
            try:
                return self._call_model_once(messages)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts or not self._is_retryable_error(exc):
                    raise
                delay = min(60.0, 2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                print(
                    f"{self.name} API call failed with {exc.__class__.__name__}; "
                    f"retrying in {delay:.1f}s ({attempt}/{attempts - 1})",
                    flush=True,
                )
                time.sleep(delay)

        raise RuntimeError("Model call failed without raising a retryable exception.") from last_error

    def _call_model_once(self, messages: list[dict[str, str]]) -> Any:
        try:
            from openai import OpenAI
        except ImportError:
            return self._call_legacy_openai(messages)

        client = OpenAI(api_key=self.api_key, base_url=self.api_base, max_retries=0)
        return client.chat.completions.create(
            model=self.model_name,
            timeout=self.timeout,
            messages=messages,
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        try:
            from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        except ImportError:
            APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = ()  # type: ignore[assignment]

        retryable_openai_errors = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        if isinstance(exc, retryable_openai_errors):
            return True

        try:
            import requests
        except ImportError:
            requests = None
        if requests is not None and isinstance(exc, (requests.ConnectionError, requests.Timeout)):
            return True

        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "unexpected_eof_while_reading",
                "connection error",
                "connection reset",
                "temporarily unavailable",
                "timeout",
                "too many requests",
                "status_code: 429",
                "status_code: 500",
                "status_code: 502",
                "status_code: 503",
                "status_code: 504",
            )
        )

    def _call_legacy_openai(self, messages: list[dict[str, str]]) -> Any:
        import openai

        if not hasattr(openai, "ChatCompletion"):
            return self._call_http_chat_completions(messages)

        openai.api_key = self.api_key
        openai.api_base = self.api_base
        response = openai.ChatCompletion.create(
            model=self.model_name,
            request_timeout=self.timeout,
            messages=messages,
        )
        choices = [
            SimpleNamespace(
                finish_reason=choice.get("finish_reason"),
                message=SimpleNamespace(content=choice.get("message", {}).get("content", "")),
            )
            for choice in response.get("choices", [])
        ]
        usage_data = response.get("usage")
        usage = None
        if usage_data is not None:
            usage = SimpleNamespace(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        return SimpleNamespace(choices=choices, usage=usage)

    def _call_http_chat_completions(self, messages: list[dict[str, str]]) -> Any:
        import requests

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for chat completions.")

        endpoint = self.api_base.rstrip("/") + "/chat/completions"
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
            },
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Chat completion request failed: {response.status_code} {response.text}") from exc

        payload = response.json()
        choices = [
            SimpleNamespace(
                finish_reason=choice.get("finish_reason"),
                message=SimpleNamespace(
                    content=choice.get("message", {}).get("content", ""),
                    reasoning_content=choice.get("message", {}).get("reasoning_content", ""),
                ),
            )
            for choice in payload.get("choices", [])
        ]
        usage_data = payload.get("usage") or {}
        usage = SimpleNamespace(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return SimpleNamespace(choices=choices, usage=usage)
