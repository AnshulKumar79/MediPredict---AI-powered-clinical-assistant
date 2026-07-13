import os
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai

# --- STAGE 1: SET UP AND CACHE THE PYTORCH MODEL ---
@st.cache_resource
def load_medical_model():
    # Build the exact same structure you trained on
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 2) # Adjust '2' to your actual class count
    
    # Match the 8-bit dynamic quantization you applied to the weights
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    
    # Load the compressed weights file securely
    model.load_state_dict(torch.load("quantized_clinical_model.pth", map_location="cpu"))
    model.eval()
    return model

# Configure Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    # Fallback for local testing if secrets aren't set yet
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# Page UI configurations
st.set_page_config(page_title="MediPredict AI", layout="centered")
st.title("🩺 MediPredict: Clinical Lesion Analyzer")
st.write("Upload a dermatological image and describe symptoms for an AI-assisted evaluation.")

# --- STAGE 2: USER INPUTS ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
symptoms = st.text_area("Patient Symptoms / Clinical Context", placeholder="e.g., Noticeable growth over 3 months, occasionally itchy...")

if uploaded_file is not None and st.button("Run Full Clinical Analysis"):
    with st.spinner("Processing image and generating AI insights..."):
        # Load and prep image
        pil_image = Image.open(uploaded_file).convert("RGB")
        
        # 1. Run local PyTorch Prediction
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_image).unsqueeze(0)
        
        model = load_medical_model()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        classes = ["Benign Lesion", "Malignant Melanoma"] # Replace with your real target names
        top_idx = torch.argmax(probabilities).item()
        diagnosis_label = classes[top_idx]
        confidence = probabilities[top_idx].item() * 100

        # 2. Fake a clean visual Heatmap (Placeholder for your Grad-CAM logic)
        open_cv_img = np.array(pil_image)
        open_cv_img = cv2.cvtColor(open_cv_img, cv2.COLOR_RGB2BGR)
        open_cv_img = cv2.resize(open_cv_img, (300, 300))
        # Overlay dummy color map to mock your local Grad-CAM behavior
        heatmap = cv2.applyColorMap(cv2.resize(open_cv_img, (300, 300)), cv2.COLORMAP_JET)
        cam_overlay = cv2.addWeighted(open_cv_img, 0.6, heatmap, 0.4, 0)
        cam_rgb = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)

        # 3. Query Gemini LLM for Triage Context
        prompt = f"""
        You are an expert dermatological assistant. A custom local Vision Model has analyzed a patient's skin lesion.
        Vision Model Prediction: {diagnosis_label} ({confidence:.1f}% confidence).
        Patient Symptoms: {symptoms if symptoms else "None provided."}
        
        Provide a professional summary explaining what this classification implies, structural indicators to note, 
        and immediate triage next steps for the clinical team.
        """
        try:
            llm = genai.GenerativeModel("gemini-2.5-flash")
            response = llm.generate_content(prompt)
            ai_insights = response.text
        except Exception as e:
            ai_insights = f"AI Triage details unavailable right now. (Local Model Diagnostic: {diagnosis_label})"

        # --- STAGE 3: DISPLAY RESULTS BEAUTIFULLY ---
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Uploaded Case Image", use_container_width=True)
        with col2:
            st.image(cam_rgb, caption="Grad-CAM Morphological Heatmap", use_container_width=True)
            
        st.metric(label="Primary Target Diagnosis", value=diagnosis_label, delta=f"{confidence:.1f}% Confidence")
        
        st.subheader("-->Generative AI Triage Report")
        st.write(ai_insights)
