from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    requested_model: str
    served_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    latency_ms: float
    temperature: float | None
    top_p: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(Protocol):
    def generate(self, *, instructions: str, input_text: str) -> LLMResponse: ...


class OpenAILLMClient:
    """Thin OpenAI Responses API wrapper with real usage accounting."""

    def __init__(
        self,
        client,
        model: str,
        *,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
    ) -> None:
        """Pass temperature=None/top_p=None for models that reject these
        parameters; they are then omitted and the provider default applies."""
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, *, instructions: str, input_text: str) -> LLMResponse:
        sampling = {k: v for k, v in (("temperature", self.temperature), ("top_p", self.top_p))
                    if v is not None}
        start = time.monotonic()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_text,
            **sampling,
        )
        latency_ms = (time.monotonic() - start) * 1000.0

        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "input_tokens", None) if usage else None
        out_tokens = getattr(usage, "output_tokens", None) if usage else None
        details = getattr(usage, "input_tokens_details", None) if usage else None
        cached = getattr(details, "cached_tokens", None) if details else None

        return LLMResponse(
            text=response.output_text,
            requested_model=self.model,
            served_model=getattr(response, "model", None),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
            temperature=self.temperature,
            top_p=self.top_p,
        )
