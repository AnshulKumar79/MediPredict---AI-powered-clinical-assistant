# MediPredict---AI-powered-clinical-assistant

# LIVE DEMO: https://medipredict-bay.vercel.app/
# API: https://medipredict-ai-skin-specialist.onrender.com

MediPredict AI Lite is a lightweight, high-performance web service built using FastAPI, ONNX Runtime, and Google Gemini. The application is specifically designed to run efficiently in resource-constrained environments (such as Render's 512MB RAM free tier) by bypassing heavy machine learning frameworks like PyTorch.

The system accepts a dermatological lesion image and accompanying patient clinical symptoms, processes the image using an optimized 7-class ONNX model trained on the HAM10000 dataset, and routes the classification to the Gemini API to generate an actionable, context-aware triage summary.

---

## Architecture Overview

1. **API Layer (FastAPI):** Handles image file uploads, text symptom processing, and JSON response routing with a low memory baseline footprint (~120MB idle).
2. **Inference Engine (ONNX Runtime):** Runs the trained 7-class Convolutional Neural Network (EfficientNet-B0) using matrix operations directly on the CPU without requiring a heavy PyTorch engine.
3. **Generative Triage Layer (Gemini API):** Contextualizes the classification with patient-reported data to deliver an organized clinical report.

---

## 7-Class Disease Mapping (Dataset: Skin Cancer HAM10000)

The model classifies skin lesions across the following categories:
* `0 (akiec)`: Actinic keratoses and Intraepithelial Carcinoma (Pre-cancerous)
* `1 (bcc)`: Basal Cell Carcinoma (Malignant)
* `2 (bkl)`: Benign Keratosis-like Lesions (Non-cancerous)
* `3 (df)`: Dermatofibroma (Benign skin growth)
* `4 (mel)`: Melanoma (Highly Malignant)
* `5 (nv)`: Melanocytic Nevi (Common Benign Mole)
* `6 (vasc)`: Vascular Lesions (Benign blood vessel cluster)

---

## Project Structure

```text
├── app.py                  # Main FastAPI application script
├── requirements.txt        # Python package dependencies
├── model.onnx              # Optimized ONNX model architecture graph
├── model.onnx.data         # Embedded neural network weights file
└── README.md               # Documentation
