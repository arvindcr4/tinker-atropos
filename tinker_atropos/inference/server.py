from fastapi import FastAPI
from tinker_atropos.types import CompletionRequest, CompletionResponse, UpdateWeightsRequest

app = FastAPI(title="Tinker Inference Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    # TODO: implement
    raise NotImplementedError()


@app.post("/internal/update_weights")
async def update_weights(request: UpdateWeightsRequest):
    # TODO: implement
    raise NotImplementedError()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
