"""Conversation history with sliding window and system prompt support."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "thinking"
    content: str
    timestamp: float = 0.0


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    max_turns: int = 20
    _turn_count: int = field(default=0, init=False)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        if role == "assistant":
            self._turn_count += 1
        self._trim()

    def set_system(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.messages = [m for m in self.messages if m.role != "system"]
        if prompt:
            self.messages.insert(0, Message(role="system", content=prompt))

    def clear(self) -> None:
        self.messages = []
        self._turn_count = 0
        if self.system_prompt:
            self.messages.append(Message(role="system", content=self.system_prompt))

    def _trim(self) -> None:
        while self._turn_count > self.max_turns:
            if self.messages and self.messages[0].role == "system":
                self.messages.pop(1)
            elif self.messages:
                self.messages.pop(0)
            self._turn_count = max(0, self._turn_count - 1)

    def to_prompt(self, tokenizer=None, enable_thinking: bool = False) -> str:
        if tokenizer is not None:
            return self._to_chatml(tokenizer, enable_thinking)
        return self._to_text(enable_thinking)

    def _to_chatml(self, tokenizer, enable_thinking: bool) -> str:
        formatted = []
        for msg in self.messages:
            if msg.role == "system":
                formatted.append(f"<|system|>\n{msg.content}</s>")
            elif msg.role == "user":
                formatted.append(f"<|user|>\n{msg.content}</s>")
            elif msg.role == "assistant":
                formatted.append(f"<|assistant|>\n{msg.content}</s>")
            elif msg.role == "thinking":
                formatted.append(f"<|thinking|>\n{msg.content}</s>")
        if enable_thinking:
            formatted.append("<|thinking|>\n")
        else:
            formatted.append("<|assistant|>\n")
        return "\n".join(formatted)

    def _to_text(self, enable_thinking: bool) -> str:
        parts = []
        for msg in self.messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}\n\n")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}\n")
            elif msg.role == "thinking":
                parts.append(f"[Thinking: {msg.content}]\n")
        suffix = "Assistant:" if not enable_thinking else "[Thinking]:"
        parts.append(suffix)
        return "\n".join(parts)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def token_estimate(self) -> int:
        return sum(len(m.content.split()) * 1.3 for m in self.messages)
