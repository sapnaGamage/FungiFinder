from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL = load_model("model/all_mushroom_classifier.keras", compile=False)
IMAGE_SIZE = (224, 224)
X_OFFSET = float(os.environ.get("X_OFFSET", "-0.08753"))

CLASS_NAMES = [
    'Agaricus bisporus', 'Auricularia polytricha', 'Lentinus edodes',
    'Lentinus giganteus', 'Phallus indusiatus', 'Pleurotus cystidiosus',
    'Pleurotus ostreatus', 'Schizophyllum commune',
    'Termitomyces eurrhizus', 'Volvariella volvacea'
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) + X_OFFSET
    return {
        'predicted_class': predicted_class,
        'confidence': confidence
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8090)
