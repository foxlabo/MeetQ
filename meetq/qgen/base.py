"""LLM provider interface and the question data model."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

QUESTION_TYPES = ("clarify", "unresolved", "contradiction", "risk", "comprehension")
MODES = ("followup", "comprehension")


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests


@dataclass
class GeneratedQuestion:
    text: str
    type: str
    priority: int
    evidence_quote: str
    evidence_start_sec: float = 0.0
    evidence_segment_ids: list[int] = field(default_factory=list)


class LLMProvider(abc.ABC):
    """Returns a dict validated against `schema` plus the token usage of the call."""

    name = "base"

    @abc.abstractmethod
    def generate_json(
        self, system: str, user: str, schema: dict, schema_name: str
    ) -> tuple[dict, TokenUsage]: ...


def question_schema() -> dict:
    """Strict JSON Schema for Structured Outputs (all properties required, no extras)."""
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "会議参加者に投げる具体的な質問文",
                        },
                        "type": {"type": "string", "enum": list(QUESTION_TYPES)},
                        "priority": {
                            "type": "integer",
                            "description": "1 が最優先、3 が最低。1..3 の整数",
                        },
                        "evidence_quote": {
                            "type": "string",
                            "description": "文字起こしからの逐語引用。必ず原文と完全一致させる",
                        },
                    },
                    "required": ["text", "type", "priority", "evidence_quote"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }
