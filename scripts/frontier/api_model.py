"""Unified API client wrapper for frontier model evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ApiModel:
    """Minimal Model-compatible wrapper for OpenAI and Anthropic chat APIs."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        load_env_file()
        self.provider = provider.lower()
        self.model = model
        self.name = f"{provider}:{model}"
        self.temperature = temperature
        self.max_retries = max_retries
        if self.provider == "anthropic":
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
            self._client = Anthropic(api_key=api_key)
        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in .env")
            self._client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def is_reasoning_model(self) -> bool:
        """OpenAI o-series models allocate tokens to internal reasoning before visible text."""
        return self.provider == "openai" and self.model.startswith("o")

    def chat_prompt(self, user_text: str, assistant_prefill: str = "") -> str:
        self._user_text = user_text
        self._assistant_prefill = assistant_prefill
        return user_text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float | None = None,
        restrict_first_token_to: list[str] | None = None,
        assistant_prefill: str = "",
    ) -> str:
        del restrict_first_token_to, prompt
        user_text = getattr(self, "_user_text", "")
        prefill = assistant_prefill or getattr(self, "_assistant_prefill", "")
        # Anthropic reasoning models reject assistant prefill; fold into user turn.
        if self.provider == "anthropic" and prefill:
            user_text = user_text.rstrip() + f"\n\nBegin your reply with:\n{prefill.strip()}"
            prefill = ""
        temp = self.temperature if temperature is None else temperature
        if do_sample and temp == 0.0:
            temp = 0.7
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    return self._generate_anthropic(user_text, prefill, max_new_tokens, temp)
                return self._generate_openai(user_text, prefill, max_new_tokens, temp, do_sample)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"API generation failed after {self.max_retries} retries") from last_err

    def _generate_anthropic(
        self, user_text: str, assistant_prefill: str, max_tokens: int, temperature: float
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_text}],
        )
        if assistant_prefill:
            kwargs["messages"].append({"role": "assistant", "content": assistant_prefill.rstrip()})
        msg = self._client.messages.create(**kwargs)
        parts = [block.text for block in msg.content if hasattr(block, "text")]
        text = "".join(parts)
        return text

    def _generate_openai(
        self,
        user_text: str,
        assistant_prefill: str,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> str:
        messages: list[dict] = [{"role": "user", "content": user_text}]
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})
        # Reasoning models (o-series) use max_completion_tokens
        is_reasoning = self.is_reasoning_model
        kwargs: dict = dict(model=self.model, messages=messages)
        if is_reasoning:
            # Reasoning models consume most of the budget on hidden chain-of-thought.
            kwargs["max_completion_tokens"] = max(max_tokens, 4096)
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature if do_sample else 0.0
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        return (msg.content or "").strip()
