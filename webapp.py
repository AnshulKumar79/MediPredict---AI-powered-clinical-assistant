import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai

# --- CLASS FOR GENUINE GRAD-CAM ---
class EfficientNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Target the final convolutional feature extractor block of EfficientNet-B0
        self.target_layer = self.model.features[-1]
        
        # Register hooks using modern full backward protocol to secure the tracking maps
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        target_score = output[0][class_idx]
        target_score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.clamp(cam, min=0)
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        heatmap = cam.squeeze().cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap

# --- CACHE THE HEAVY STANDARD FLOAT32 MODEL ---
@st.cache_resource
def load_medical_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    
    # CRUCIAL: Set to 7 classes to match your dataset mapping
    model.classifier[1] = nn.Linear(num_features, 7)
    
    model.load_state_dict(torch.load("best_clinical_model_unfrozen.pth", map_location="cpu"))
    model.eval()
    return model

# Configure Gemini Authentication
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# Setup Streamlit Interface
st.set_page_config(page_title="MediPredict AI", layout="centered")
st.title("🩺 MediPredict: 7-Class Lesion Analyzer")
st.write("Upload a clinical skin image and outline symptoms for unified diagnostic prediction and attention mapping.")

# --- CORE USER SELECTIONS ---
uploaded_file = st.file_uploader("Choose a lesion image...", type=["jpg", "jpeg", "png"])
symptoms = st.text_area("Patient Symptoms / Clinical Context", placeholder="e.g., Growing rapidly over 2 months, rough borders, bleeds easily...")

if uploaded_file is not None and st.button("Execute Full Diagnostic Sequence"):
    with st.spinner("Analyzing image features and pulling clinical context framework..."):
        
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_w, orig_h = pil_image.size
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_image).unsqueeze(0)
        input_tensor.requires_grad_()

        model = load_medical_model()
        cam_pipeline = EfficientNetGradCAM(model)
        
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # YOUR EXACT HAM10000 DATASET DISEASE MAPPING
        disease_map = {
            0: ('akiec', "Actinic keratoses and Intraepithelial Carcinoma (Pre-cancerous)"),
            1: ('bcc', "Basal Cell Carcinoma (Malignant)"),
            2: ('bkl', "Benign Keratosis-like Lesions (Non-cancerous)"),
            3: ('df', "Dermatofibroma (Benign skin growth)"),
            4: ('mel', "Melanoma (Highly Malignant)"),
            5: ('nv', "Melanocytic Nevi (Common Benign Mole)"),
            6: ('vasc', "Vascular Lesions (Benign blood vessel cluster)")
        }
        
        top_idx = torch.argmax(probabilities).item()
        short_code, diagnosis_label = disease_map[top_idx]
        confidence = probabilities[top_idx].item() * 100

        # Extract authentic Grad-CAM structural map via backward tracking
        raw_heatmap = cam_pipeline.generate_heatmap(input_tensor, top_idx)
        
        # Blend the attention overlay cleanly using OpenCV transformations
        open_cv_img = np.array(pil_image.resize((224, 224)))
        heatmap_colored = np.uint8(255 * raw_heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        
        cam_overlay = cv2.addWeighted(open_cv_img, 0.6, heatmap_colored, 0.4, 0)
        cam_final = cv2.resize(cam_overlay, (orig_w, orig_h))

        # Build prompt and query Gemini API with full medical names
        prompt = f"""
        You are an expert clinical dermatological companion. A localized deep learning CNN has categorized a patient's skin lesion.
        
        Model Output Classification: {diagnosis_label} (Short code: {short_code})
        Calculated Prediction Confidence: {confidence:.2f}%
        Reported Patient Symptoms: {symptoms if symptoms else "None noted."}
        
        Provide an organized clinical context evaluation explaining what these visual details typically imply, 
        morphological variables that demand immediate evaluation, and recommended secondary routing procedures.
        """
        try:
            llm = genai.GenerativeModel("gemini-2.5-flash")
            response = llm.generate_content(prompt)
            ai_insights = response.text
        except Exception as e:
            ai_insights = f"Generative AI Context layer currently offline. (Vision Model output safely parsed: {diagnosis_label} with {confidence:.1f}% confidence)."

        # --- DISPLAY RESULTS ---
        st.success("Full Sequence Diagnostic Pipeline Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Original Input Case Image", use_container_width=True)
        with col2:
            st.image(cam_final, caption="True Backpropagation Activation Heatmap", use_container_width=True)
            
        st.metric(label="Calculated Model Assessment", value=diagnosis_label, delta=f"{confidence:.2f}% Confidence Score")
        
        st.subheader("-->Generative AI Clinical Triage Protocol")
        st.write(ai_insights)
