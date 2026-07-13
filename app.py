import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv


load_dotenv()

# Grad-CAM Tooling
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Google Gemini API SDK
import google.generativeai as genai

app = FastAPI(
    title="MediPredict_API",
    description="Open triage backend combining PyTorch Vision, Grad-CAM, and Gemini LLM Clinical Synthesis.",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all web origins to make requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hardware Environment (CPU optimized for web tasks)
device = torch.device("cpu")

# Securely configure Gemini SDK using system environment configurations
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    print("Gemini API key not found")

#NEURAL NETWORK ARCHITECTURE
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 7)


WEIGHTS_FILE = 'best_clinical_model_unfrozen.pth'
if os.path.exists(WEIGHTS_FILE):
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"EfficientNet framework initialized with weights: {WEIGHTS_FILE}")
else:
    print(f"ERROR: Weights file '{WEIGHTS_FILE}' not found")

# Prep specific Conv layers for structural tracking map extraction
for param in model.features[-1].parameters():
    param.requires_grad = True

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Standardize incoming visual formats to match training matrix parameters
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

disease_map = {
    0: ('akiec', "Actinic Keratoses / Intraepithelial Carcinoma (Pre-cancerous)"),
    1: ('bcc', "Basal Cell Carcinoma (Malignant skin cancer)"),
    2: ('bkl', "Benign Keratosis-like Lesions (Non-cancerous)"),
    3: ('df', "Dermatofibroma (Benign firm nodule)"),
    4: ('mel', "Melanoma (Highly Malignant / Aggressive skin cancer)"),
    5: ('nv', "Melanocytic Nevi (Common Benign Mole)"),
    6: ('vasc', "Vascular Lesions (Benign vascular cluster)")
}

#API Endpoint
@app.post("/analyze")
async def analyze_lesion_endpoint(
    file: UploadFile = File(...),
    symptoms: str = Form(...)  # Direct, open injection parsing
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The submitted asset must be a valid image format.")
    
    try:
        # Phase 1: Image Stream Deconstruction
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(raw_image).unsqueeze(0).to(device)
        
        # Phase 2: Core Vision Model Categorization
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        class_id = predicted_idx.item()
        short_code, diagnosis_label = disease_map[class_id]
        confidence_pct = round(confidence.item() * 100, 2)
        
        # Phase 3: Spatial Heatmap Generation via Grad-CAM
        targets = [ClassifierOutputTarget(class_id)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Reconstruct standard RGB backgrounds from un-normalized data slices
        img_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_img = np.clip(std * img_np + mean, 0, 1)
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        heatmap_pil = Image.fromarray((cam_image * 255).astype(np.uint8))
        
        buffered = io.BytesIO()
        heatmap_pil.save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Phase 4: Text Synthesis Generation via Gemini
        prescribed_actions = "Gemini interface configuration inactive. Consult a primary healthcare facility immediately."
        
        if GEMINI_KEY:
            try:
                # Utilizing the Gemini 3.5 Flash architecture
                llm_model = genai.GenerativeModel('gemini-3.5-flash')
                
                prompt = f"""
                You are a professional medical triage support agent assisting with automated pre-screen checks.
                An image analysis model evaluated a skin lesion and flagged:
                - Suspected Classification: {diagnosis_label} ({short_code.upper()})
                - Network Prediction Confidence: {confidence_pct}%
                
                The patient provided the following symptom context:
                "{symptoms}"
                
                Compose a highly structured triage summary for the patient containing:
                1. Context Analysis: Explain what the label implies in standard, non-alarmist terms.
                2. Direct Symptoms Correlation: Connect their explicit self-reported description to the finding.
                3. Prescribed Actions: Detail concrete next steps (monitoring guidelines, appointment advice).
                4. Medical Disclaimer: State clearly that this automated assistant is not a replacement for a clinical treatment.
                
                Ensure the style is professional, easy to digest, and clean.
                """
                
                response = llm_model.generate_content(prompt)
                prescribed_actions = response.text
                
            except Exception as gemini_err:
                prescribed_actions = f"Gemini system experienced an inline handling error: {str(gemini_err)}"

        # Phase 5: Structured Return Package Data Object
        return {
            "status": "success",
            "vision_insights": {
                "detected_class_code": short_code.upper(),
                "diagnosis_description": diagnosis_label,
                "ai_confidence_percentage": confidence_pct
            },
            "explainable_ai": {
                "heatmap_format": "image/jpeg",
                "heatmap_image_base64": heatmap_base64
            },
            "clinical_triage": {
                "user_reported_symptoms_received": symptoms,
                "gemini_prescribed_actions": prescribed_actions
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {str(e)}")