from fastapi import FastAPI
from pydantic import BaseModel
from model import analyze_bio, llm_rewrite
import numpy as np

app = FastAPI(title="AI Green/Red Flag Detector", version="1.0")



def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


# Request models
class BioRequest(BaseModel):
    text: str


class RewriteRequest(BaseModel):
    text: str
    target_tone: str = "healthy & respectful"


@app.post("/classify")
def classify_bio(req: BioRequest):
    flags, scores = analyze_bio(req.text)
    return to_python({
        "scores": scores,
        "flags": flags,
    })


@app.post("/rewrite")
def rewrite_bio(req: RewriteRequest):
    flags, _ = analyze_bio(req.text)
    rewritten = llm_rewrite(req.text, req.target_tone, flags)
    return {
        "improved_bio": rewritten,
        "notes": "Red/yellow flags removed or softened",
    }
