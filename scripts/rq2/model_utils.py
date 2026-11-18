"""Lightweight HF model wrapper for metacognition eval."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RQ1_DIR = Path(__file__).resolve().parents[1] / "rq1"
if str(RQ1_DIR) not in sys.path:
    sys.path.append(str(RQ1_DIR))

from moe_lora import load_moe_lora  # noqa: E402


def letter_token_ids(tokenizer, letters: list[str]) -> list[int]:
    """Token ids that decode to a valid MCQ option letter as the first generated token."""
    valid = set(letters)
    found: set[int] = set()
    for letter in letters:
        for text in (letter, f" {letter}", f"({letter})", f"{letter}.", f"{letter},", f"{letter})"):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                found.add(encoded[0])
    if not found:
        for tid in range(tokenizer.vocab_size):
            piece = tokenizer.decode([tid]).strip()
            if piece in valid:
                found.add(tid)
    return sorted(found)


def _first_token_fn(prompt_len: int, allowed_ids: list[int], vocab_size: int):
    all_ids = list(range(vocab_size))

    def fn(_batch_id: int, input_ids: torch.LongTensor) -> list[int]:
        if len(input_ids) == prompt_len:
            return allowed_ids
        return all_ids

    return fn


def is_o1_style_model(model_name: str) -> bool:
    lower = model_name.lower()
    return "huatuogpt" in lower or "m1-" in lower or "huatuo" in lower or "/m1" in lower


def thinking_prefill(model_name: str) -> str:
    """Assistant prefill that anchors o1-style models in the ## Thinking section."""
    if is_o1_style_model(model_name):
        return "## Thinking\n"
    return ""


def consistency_answer_prefill(model_name: str) -> str:
    """Assistant prefill that matches o1-style medical models' force-answer format."""
    if is_o1_style_model(model_name):
        return "## Final Response\nAnswer: "
    return "Answer: "


def default_max_new_tokens(model_name: str, base: int = 1024) -> int:
    """m1 often needs a larger budget to reach ## Final Response after long CoT."""
    lower = model_name.lower()
    if "m1-" in lower or "/m1" in lower:
        return max(base, 4096)
    if is_o1_style_model(model_name):
        return max(base, 2048)
    return base


class Model:
    def __init__(
        self,
        name: str,
        load_in_4bit: bool = False,
        base_model: str | None = None,
        adapter_dir: str | None = None,
        adapter_type: str = "none",
    ):
        self.name = name
        self.base_model = base_model or name
        self.adapter_dir = adapter_dir
        self.adapter_type = adapter_type

        kwargs: dict = dict(trust_remote_code=True, device_map="auto")
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        self.tok = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.lm = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        if adapter_dir:
            if adapter_type == "dense":
                self.lm = PeftModel.from_pretrained(self.lm, adapter_dir)
            elif adapter_type == "moe":
                self.lm = load_moe_lora(self.lm, adapter_dir)
            else:
                raise ValueError(f"Unknown adapter_type: {adapter_type}")
        self.lm.eval()

    def chat_prompt(self, user_text: str, assistant_prefill: str = "") -> str:
        messages = [{"role": "user", "content": user_text}]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if assistant_prefill:
            prompt = prompt + assistant_prefill
        return prompt

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 0.7,
        restrict_first_token_to: list[str] | None = None,
    ) -> str:
        inputs = self.tok(prompt, return_tensors="pt").to(self.lm.device)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            pad_token_id=self.tok.eos_token_id,
        )
        if restrict_first_token_to:
            allowed = letter_token_ids(self.tok, restrict_first_token_to)
            if allowed:
                gen_kwargs["prefix_allowed_tokens_fn"] = _first_token_fn(
                    prompt_len, allowed, self.tok.vocab_size
                )
        out = self.lm.generate(**inputs, **gen_kwargs)
        gen = out[0][prompt_len:]
        return self.tok.decode(gen, skip_special_tokens=True)
