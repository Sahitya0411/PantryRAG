"""Pluggable LLM backends.

  PANTRYRAG_LLM=hf          local open model via transformers (default:
                            Qwen/Qwen2.5-1.5B-Instruct; override PANTRYRAG_HF_MODEL)
  PANTRYRAG_LLM=anthropic   Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY;
                            model override PANTRYRAG_CLAUDE_MODEL)
  PANTRYRAG_LLM=extractive  no LLM: deterministic template over retrieved recipes
  PANTRYRAG_LLM=auto        anthropic if a key is set, else hf (default)
"""
from __future__ import annotations

import os
from typing import Protocol

Message = dict  # {"role": "user"|"assistant", "content": str}


class LLM(Protocol):
    """Interface every backend implements: a `name` and a `chat()` method that takes a
    system prompt plus chat messages and returns the reply text."""
    name: str

    def chat(self, system: str, messages: list[Message], max_tokens: int = 700) -> str: ...


class HFLocalLLM:
    """Local open-weights chat model run with Hugging Face transformers (GPU/MPS if available)."""

    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id or os.environ.get("PANTRYRAG_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.name = f"hf:{self.model_id}"
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        # float16 on Apple MPS degenerates into repetition loops on long RAG prompts;
        # float32 is stable (1.5B params = ~6 GB, fine on a 16 GB Mac).
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype).to(self.device)
        self.model.eval()
        self._torch = torch

    def chat(self, system: str, messages: list[Message], max_tokens: int = 700) -> str:
        """Generate a reply with greedy decoding and return only the new text."""
        msgs = [{"role": "system", "content": system}] + messages
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        # greedy decoding (do_sample=False) keeps evaluation runs reproducible;
        # a mild repetition penalty stops the small model from looping
        with self._torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None,
                                      repetition_penalty=1.05,
                                      pad_token_id=self.tok.eos_token_id)
        # generate() returns prompt + completion; drop the first n prompt tokens
        n = enc["input_ids"].shape[1]
        return self.tok.decode(out[0, n:], skip_special_tokens=True).strip()


class AnthropicLLM:
    """Hosted model accessed through the Anthropic SDK (needs an API key in the environment)."""

    def __init__(self, model: str | None = None):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("PANTRYRAG_CLAUDE_MODEL", "claude-opus-5-5")
        self.name = f"anthropic:{self.model}"

    def chat(self, system: str, messages: list[Message], max_tokens: int = 4000) -> str:
        """Send one request and return the concatenated text blocks of the reply."""
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                      output_config={"effort": "low"})
        try:
            # server-side refusal fallback (routes a declined request to another model)
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except (TypeError, self.anthropic.BadRequestError):
            # older SDK / account without the beta: fall back to the standard endpoint
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class ExtractiveLLM:
    """Placeholder backend; generation.py renders a template instead of calling it."""
    name = "extractive"

    def chat(self, system: str, messages: list[Message], max_tokens: int = 700) -> str:
        raise RuntimeError("extractive backend has no free-form chat")


# one instance per backend, so a local model is loaded into memory only once
_CACHE: dict[str, LLM] = {}


def get_llm(kind: str | None = None) -> LLM:
    """Return the backend named by `kind` (or $PANTRYRAG_LLM): hf, anthropic,
    extractive or auto. Instances are cached and reused."""
    kind = (kind or os.environ.get("PANTRYRAG_LLM", "auto")).lower()
    if kind == "auto":
        kind = "anthropic" if (os.environ.get("ANTHROPIC_API_KEY")
                               or os.environ.get("ANTHROPIC_AUTH_TOKEN")) else "hf"
    if kind not in _CACHE:
        _CACHE[kind] = {"hf": HFLocalLLM, "anthropic": AnthropicLLM,
                        "extractive": ExtractiveLLM}[kind]()
    return _CACHE[kind]
