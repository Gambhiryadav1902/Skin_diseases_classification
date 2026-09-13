from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import shutil
import os

from src.models.predict import predict_multiple

router = APIRouter()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.get("/")
def home():
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return {"message": "Skin Disease API Running"}



@router.post("/predict")
def predict_api(files: List[UploadFile] = File(...)):
    file_paths = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    label, confidence = predict_multiple(file_paths)

    # Clean up uploaded temporary files
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "num_files": len(file_paths)
    }