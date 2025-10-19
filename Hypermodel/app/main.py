import os, time, uvicorn
from fastapi import FastAPI, HTTPException
from .schemas import PredictIn, PredictOut
from .service import route_and_predict
from .loaders.common import DEVICE

app = FastAPI(title="HyperModel (AR + EN)", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    t0 = time.time()
    try:
        res = route_and_predict(payload.text, payload.lang)
        return PredictOut(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference_error: {e}") from e
    finally:
        _ = time.time() - t0

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
