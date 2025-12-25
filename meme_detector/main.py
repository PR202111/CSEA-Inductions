from fastapi import FastAPI, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io
import hashlib
from query import process_meme, process_query


MEME_CACHE = {}

app = FastAPI(title="AI Meme Detector", version="1.0")


def get_meme_id(image_bytes: bytes) -> str:
    """Generate a unique ID for a meme based on its bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


def process_and_cache_meme(image_bytes: bytes):
    """Process a meme image, cache the results, and return the ID + results."""
    meme_id = get_meme_id(image_bytes)

    if meme_id in MEME_CACHE:
        return meme_id, MEME_CACHE[meme_id]


    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)


    result = process_meme(img_array)


    MEME_CACHE[meme_id] = result
    return meme_id, result


@app.post("/meme-read")
async def meme_read(file: UploadFile):
    """Upload an image and extract text + detected template."""
    image_bytes = await file.read()
    meme_id, result = process_and_cache_meme(image_bytes)

    return {
        "meme_id": meme_id,
        "extracted_text": result.get("text", ""),
        "template": result.get("detected_template", "")
    }


@app.post("/meme-emotion")
async def meme_emotion(meme_id: str):
    """Return sentiment/emotion analysis for a cached meme."""
    if meme_id not in MEME_CACHE:
        raise HTTPException(status_code=404, detail="Invalid meme_id")

    cached = MEME_CACHE[meme_id]
    return {
        "sentiment": cached.get("current_vibe", ""),
        "confidence": cached.get("label_confidence", 0.0)
    }


@app.post("/meme-rewrite")
async def meme_rewrite(meme_id: str):
    """Generate improved/rephrased text for a meme based on analysis."""
    if meme_id not in MEME_CACHE:
        raise HTTPException(status_code=404, detail="Invalid meme_id")

    cached = MEME_CACHE[meme_id]
    rewrites = process_query(cached)

    return {
        "meme_id": meme_id,
        "rewrites": rewrites
    }
