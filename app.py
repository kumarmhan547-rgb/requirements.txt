import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Medical Diagnosis Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI-Powered Medical Diagnosis Chatbot")
st.caption("Multi-Specialist Medical Scanner & Health Insights (Bone, Lung, Eye)")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.text_input("Enter your Gemini API Key:", type="password")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

doctor_prompt = """
You are a highly qualified AI Medical Assistant specializing in triage and scan analysis.
When analyzing images (X-rays, CT scans, Eye scans) or user symptoms:
1. Identify potential conditions/findings in simple terms.
2. Explain underlying causes and early lifestyle/first-aid precautions.
3. Clarify critical red-flag signs requiring immediate emergency medical care.
4. Support explanations in both English and Telugu.
Disclaimer: This is an AI screening tool and not a doctor's final prescription.
"""
llm = genai.GenerativeModel("gemini-1.5-flash", system_instruction=doctor_prompt)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo = load_yolo()

specialty = st.selectbox(
    "Select Medical Specialty / విభాగం ఎంచుకోండి:",
    [
        "Bone Fractures (Orthopedics / ఎముకల పగుళ్లు)",
        "Chest & Lung Health (Pulmonology / ఊపిరితిత్తులు)",
        "Eye Conditions (Ophthalmology / కంటి సమస్యలు)",
        "General Symptoms & Triage (సాధారణ లక్షణాలు)"
    ]
)

user_text = st.text_input(
    "Describe symptoms / మీ లక్షణాలను వివరించండి:",
    placeholder="e.g., Chest tightness with persistent cough..."
)

uploaded_file = st.file_uploader(
    "Upload Medical Scan (X-Ray, CT, Eye Image):",
    type=["jpg", "jpeg", "png"]
)

if st.button("Analyze & Diagnose / విశ్లేషించండి"):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Scan", use_container_width=True)
        
        with st.spinner("1. Running YOLO Detection Scan..."):
            results = yolo(img)
            st.success("✅ YOLO Scan Completed.")
            
        with st.spinner("2. Generating AI Specialist Insights..."):
            prompt = (
                f"Specialty Category: {specialty}\n"
                f"Symptoms: {user_text if user_text else 'Analyze the scan for any abnormalities.'}\n"
                "Provide diagnostic breakdown, precautions, and specialist recommendations in Telugu and English."
            )
            response = llm.generate_content([prompt, img])
            st.markdown("### 📋 Medical Insights & Report:")
            st.write(response.text)
            
    elif user_text:
        with st.spinner("Generating Medical Guidance..."):
            response = llm.generate_content(
                f"Specialty: {specialty}\nSymptoms: {user_text}\n"
                "Explain causes, precautions, and doctor consultation advice in Telugu and English."
            )
            st.markdown("### 📋 Medical Guidance:")
            st.write(response.text)
    else:
        st.warning("Please upload a medical scan image or enter your symptoms.")
