from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from microlab.evals.schema import EvalTask, ModelOutput


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, task: EvalTask) -> ModelOutput:
        raise NotImplementedError


class FixtureBackend(ModelBackend):
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        text = self.answers.get(task.id, "")
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
        )


class OllamaBackend(ModelBackend):
    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout_seconds: int = 600,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def generate(self, task: EvalTask) -> ModelOutput:
        start = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": task.prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": task.max_new_tokens,
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response", "")).strip()
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
        )


class HuggingFaceCausalLMBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self._torch = torch

    def generate(self, task: EvalTask) -> ModelOutput:
        torch = self._torch
        start = time.perf_counter()
        messages = [{"role": "user", "content": task.prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = task.prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=task.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return ModelOutput(
            task_id=task.id,
            text=text,
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(completion_ids.shape[-1]),
        )


def create_backend(config: dict[str, Any]) -> ModelBackend:
    backend_type = config.get("type")
    if backend_type == "fixture":
        return FixtureBackend(dict(config.get("answers", {})))
    if backend_type == "ollama":
        return OllamaBackend(
            model=str(config["model"]),
            host=str(config.get("host", "http://127.0.0.1:11434")),
            temperature=float(config.get("temperature", 0.0)),
            timeout_seconds=int(config.get("timeout_seconds", 600)),
        )
    if backend_type == "hf_causal_lm":
        return HuggingFaceCausalLMBackend(
            model_id=str(config["model_id"]),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"unsupported backend type: {backend_type}")
