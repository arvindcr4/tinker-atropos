import os

from fastapi import FastAPI, HTTPException
from tinker_atropos.types import (
    CompletionRequest,
    CompletionResponse,
    UpdateWeightsRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from tinker_atropos.inference.wrapper import TinkerInferenceWrapper

import time
import tinker
import uuid

wrapper: TinkerInferenceWrapper | None = None
current_model_name: str = "unknown"

app = FastAPI(title="Tinker Inference Service")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@app.on_event("startup")
async def startup():
    global wrapper, current_model_name

    # TODO: These should come from a top level config
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    current_model_name = base_model

    print(f"Initializing Tinker inference service with base model: {base_model}")
    service_client = tinker.ServiceClient()
    wrapper = TinkerInferenceWrapper(service_client, base_model)
    print("Inference service ready")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": current_model_name,
        "wrapper_initialized": wrapper is not None,
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Wrapper not initialized")

    try:
        completions_list, logprobs = await wrapper.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            num_samples=request.n,
        )

        choices = [
            {
                "text": completion,
                "index": i,
                "finish_reason": "stop",
                "logprobs": logprobs[i],
            }
            for i, completion in enumerate(completions_list)
        ]

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:24]}",
            choices=choices,
            created=int(time.time()),
            model=current_model_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Wrapper not initialized")

    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = wrapper.messages_to_prompt(messages_dict)

        completions_list, logprobs = await wrapper.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            num_samples=request.n,
        )

        choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": completion,
                },
                "index": i,
                "finish_reason": "stop",
                "logprobs": logprobs[i],
            }
            for i, completion in enumerate(completions_list)
        ]

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            choices=choices,
            created=int(time.time()),
            model=current_model_name,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@app.post("/internal/update_weights")
async def update_weights(request: UpdateWeightsRequest):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Wrapper not initialized")

    try:
        await wrapper.update_weights(request.model_path)
        return {
            "status": "success",
            "model_path": request.model_path,
            "step": request.step,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weight update failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
