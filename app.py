
import streamlit as st
from PIL import Image
import torch
from timm import create_model
import joblib
import numpy as np
import torchvision.transforms as transforms

@st.cache_resource
def load_models():
    model = create_model('efficientnet_b0', pretrained=False)
    model.classifier = torch.nn.Identity()
    model.load_state_dict(torch.load("saved_models/efficientnet_b0_best.pth", map_location='cpu'))
    model.eval()

    svm = joblib.load("saved_models/svm_classifier.joblib")
    scaler = joblib.load("saved_models/standard_scaler.joblib")
    return model, svm, scaler

model, svm_classifier, scaler = load_models()

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

st.title(" Image Tamper Detection (SVM + CNN)")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        input_tensor = preprocess(image)
        with torch.no_grad():
            features = model(input_tensor).cpu().numpy()
        features_scaled = scaler.transform(features)
        prediction = svm_classifier.predict(features_scaled)[0]

    label = " Authentic" if prediction == 0 else " Tampered"
    st.markdown(f"### Prediction: {label}")
