import os
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, Header, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
import cv2
import dlib
import numpy as np

app = FastAPI()
load_dotenv()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

API_KEY = os.getenv("METAMIEJSKIE_FACE_RECOGNITION")
def get_api_key(x_api_key: str = Header(...)):
    # Compare the provided key with the key from the .env file
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing X-API-Key",
        )
    return x_api_key
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/upload-image/")
async def upload_image(api_key: str = Depends(get_api_key),image: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(os.getcwd(), image.filename)
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await image.read())

    img = cv2.imread(temp_file_path)
    if img is None:
        return JSONResponse(content={"error": "Could not read the image."}, status_code=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    data = []

    for face in faces:
        shape = predictor(gray, face)
        recognised_face_embedding = recognizer.compute_face_descriptor(img, shape)
        embedding_str = ",".join(map(str, recognised_face_embedding))
        data.append(embedding_str)
        continue
    os.remove(temp_file_path)

    return JSONResponse(content={
        "count": len(data),
        "faces": data
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)