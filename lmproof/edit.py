from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Span:
    start: int
    end: int

@dataclass(eq=True, frozen=True)
class Edit:
    span: Span
    text: str