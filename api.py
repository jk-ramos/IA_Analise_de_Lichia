from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from pymongo import MongoClient
from datetime import datetime

app = FastAPI()

# -----------------------------
# CONEXÃO COM MONGODB ATLAS
# -----------------------------
MONGO_URI = "mongodb+srv://jkramos:ocDnGF2Llqw56E2k@mobo.eswkbcg.mongodb.net/ia_lichia_db?retryWrites=true&w=majority&appName=mobo"
client = MongoClient(MONGO_URI)

db = client["ia_lichia_db"]
analises_collection = db["analises"]

# -----------------------------
# CARREGAR MODELO
# -----------------------------
MODEL_PATH = "ia_lichia/modelo_lichia.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "API de IA de Lichia funcionando com Atlas"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction >= 0.5:
            predicted_class = "nao_madura"
            confidence = prediction
        else:
            predicted_class = "madura"
            confidence = 1 - prediction

        resultado = {
            "nome_arquivo": file.filename,
            "classe_prevista": predicted_class,
            "confianca": round(float(confidence) * 100, 2),
            "data_analise": datetime.now()
        }

        analises_collection.insert_one(resultado)

        return {
            "classe_prevista": resultado["classe_prevista"],
            "confianca": resultado["confianca"],
            "nome_arquivo": resultado["nome_arquivo"],
            "mensagem": "Análise salva no MongoDB Atlas com sucesso"
        }

    except Exception as e:
        return {"erro": str(e)}