# server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import nltk
import numpy as np
import cv2
import base64
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# download punkt if missing (runs once)
nltk.download("punkt")

app = FastAPI()

# Allow requests from Flutter apps (dev). In prod restrict origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel

class TextPayload(BaseModel):
    text: str

@app.post("/spacy-tokenize")
def spacy_tokenize(payload: TextPayload):
    doc = nlp(payload.text)
    tokens = [token.text for token in doc]
    return {"tokens": tokens}

@app.get("/tokenize")
def tokenize(text: str):
    tokens = nltk.word_tokenize(text)
    return {"tokens": tokens}

@app.get("/sum-array")
def sum_array(numbers: str):
    # expect "1,2,3"
    arr = np.array([float(x) for x in numbers.split(",") if x.strip() != ""])
    return {"sum": float(np.sum(arr))}

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # do a simple op: convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, encoded = cv2.imencode('.jpg', gray)
    b64 = base64.b64encode(encoded.tobytes()).decode('ascii')
    return {"image_base64": b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
