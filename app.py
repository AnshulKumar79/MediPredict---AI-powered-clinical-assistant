import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import google.generativeai as genai

# Initialize FastAPI
app = FastAPI(title="MediPredict AI Lite", version="2.0")

# Enable CORS so your frontend can communicate with it smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# Load ONNX session globally on startup to conserve memory
# This uses ~100MB of RAM compared to PyTorch's ~350MB+ baseline
try:
    session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

# HAM10000 Dataset Target Mapping
DISEASE_MAP = {
    0: ('akiec', "Actinic keratoses and Intraepithelial Carcinoma (Pre-cancerous)"),
    1: ('bcc', "Basal Cell Carcinoma (Malignant)"),
    2: ('bkl', "Benign Keratosis-like Lesions (Non-cancerous)"),
    3: ('df', "Dermatofibroma (Benign skin growth)"),
    4: ('mel', "Melanoma (Highly Malignant)"),
    5: ('nv', "Melanocytic Nevi (Common Benign Mole)"),
    6: ('vasc', "Vascular Lesions (Benign blood vessel cluster)")
}

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": session is not None}

@app.post("/api/v2/analyze-lesion")
async def analyze_lesion(
    image: UploadFile = File(...),
    symptoms: str = Form("None provided.")
):
    if not session:
        raise HTTPException(status_code=500, detail="ONNX Model Engine is unavailable.")
        
    try:
        # 1. Read the incoming image file directly into memory bytes
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Manual Matrix Preprocessing (Zero PyTorch dependencies)
        img_resized = pil_image.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Standard ImageNet normalization math
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_array - mean) / std
        
        # Rearrange dims: HWC to CHW format & add batch axis -> (1, 3, 224, 224)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        # 3. Run Inference via ONNX
        onnx_inputs = {session.get_inputs()[0].name: input_tensor}
        onnx_outputs = session.run(None, onnx_inputs)
        raw_scores = onnx_outputs[0][0]
        
        # Stable numerical Softmax calculation
        exp_shifted = np.exp(raw_scores - np.max(raw_scores))
        probabilities = exp_shifted / exp_shifted.sum()
        
        # 4. Extract Classification Labels
        top_idx = int(np.argmax(probabilities))
        short_code, diagnosis_label = DISEASE_MAP[top_idx]
        confidence = float(probabilities[top_idx]) * 100

        # 5. Connect Triage Synthesis Layer via Gemini API
        prompt = f"""
        You are an expert clinical dermatological companion. A localized deep learning ONNX model has categorized a patient's skin lesion.
        
        Model Output Classification: {diagnosis_label} (Short code: {short_code})
        Calculated Prediction Confidence: {confidence:.2f}%
        Reported Patient Symptoms: {symptoms}
        
        Provide an organized clinical context evaluation explaining what these visual details typically imply, 
        morphological variables that demand immediate evaluation, and recommended secondary routing procedures.
        """
        
        try:
            llm = genai.GenerativeModel("gemini-2.5-flash")
            response = llm.generate_content(prompt)
            ai_insights = response.text
        except Exception as gemini_err:
            ai_insights = f"Generative backup context failure. Diagnostic fallback preserved: {diagnosis_label}."

        return {
            "short_code": short_code,
            "diagnosis": diagnosis_label,
            "confidence_score": f"{confidence:.2f}%",
            "triage_insights": ai_insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failure: {str(e)}")
