from typing import List, Dict, Any
from pydantic import BaseModel


# Request format for /v1/completions endpoint.
class CompletionRequest(BaseModel):
    prompt: str | List[str]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/completions endpoint.
class CompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str


# Request format for /internal/update_weights endpoint.
class UpdateWeightsRequest(BaseModel):
    model_path: str
    step: int | None = None


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


# Request format for /v1/chat/completions endpoint.
class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/chat/completions endpoint.
class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str
