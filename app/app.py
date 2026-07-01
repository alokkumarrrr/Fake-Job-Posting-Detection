import os
import sys
import base64
import joblib
import streamlit as st
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Append current directory to path so Streamlit can locate src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_text

# Configure page layout and visual theme
st.set_page_config(
    page_title="Fake Job Posting Detection System",
    page_icon="🛡️",
    layout="centered"
)

# Helper to encode background image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and encode background image
bg_path = os.path.join(os.path.dirname(__file__), "background.jpg")
bg_base64 = get_base64_image(bg_path)

# Custom premium styling mixing Paytm & PhonePe with light wavy background
# Added 3D depth shadows and hover lifts for card panels
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Montserrat:wght@600;800&family=Poppins:wght@400;500;600&display=swap');
    
    html, body, .stApp {{
        font-family: 'Poppins', sans-serif;
        color: #0f172a !important; /* High contrast dark slate text */
    }}
    
    /* Inject Wavy Background Image */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
    }}
    
    /* Style for markdown text elements */
    .stMarkdown p, .stMarkdown li, .stMarkdown h3, .stMarkdown h4, .stMarkdown span {{
        color: #0f172a !important;
    }}
    
    /* 3D Enhanced Title Banner */
    .brand-banner {{
        background: linear-gradient(135deg, #5f259f 0%, #00baf2 100%);
        padding: 35px 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(95, 37, 159, 0.2), 0 6px 6px rgba(95, 37, 159, 0.1);
        transform: perspective(1px) translateZ(0);
        transition: all 0.3s ease;
    }}
    
    .brand-banner:hover {{
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(95, 37, 159, 0.3), 0 10px 10px rgba(95, 37, 159, 0.15);
    }}
    
    .brand-banner h1 {{
        color: #ffffff !important;
        font-family: 'Montserrat', sans-serif;
        font-size: 2.1rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
        text-transform: uppercase;
        text-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }}
    
    .brand-banner p {{
        color: #f3e8ff !important;
        margin-top: 8px;
        font-size: 0.95rem;
        font-weight: 400;
        opacity: 0.95;
    }}
    
    /* 3D Enhanced Textarea */
    .stTextArea textarea {{
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        padding: 16px !important;
        line-height: 1.5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02), inset 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }}
    
    .stTextArea textarea:focus {{
        border-color: #5f259f !important;
        box-shadow: 0 0 0 3px rgba(95, 37, 159, 0.1), 0 6px 12px rgba(95, 37, 159, 0.05) !important;
    }}
    
    /* 3D Button Style */
    .stButton>button {{
        background: linear-gradient(90deg, #5f259f 0%, #00baf2 100%);
        color: #ffffff !important;
        border-radius: 10px;
        width: 100%;
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        box-shadow: 0 5px 10px rgba(95, 37, 159, 0.15), 0 3px 3px rgba(95, 37, 159, 0.1);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
        cursor: pointer;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(90deg, #4c1d80 0%, #0099c7 100%);
        box-shadow: 0 12px 20px rgba(95, 37, 159, 0.25), 0 6px 6px rgba(95, 37, 159, 0.15);
        transform: translateY(-2px);
    }}
    
    /* 3D Enhanced Thin Card */
    .thin-card {{
        background-color: rgba(255, 255, 255, 0.97);
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        margin-top: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.03), 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    }}
    
    .thin-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 30px rgba(0,0,0,0.08), 0 8px 12px rgba(0,0,0,0.04);
    }}
    
    .status-badge-real {{
        background-color: #dcfce7;
        color: #166534 !important;
        border: 1px solid #bbf7d0;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(22, 101, 52, 0.08);
    }}
    
    .status-badge-fake {{
        background-color: #fee2e2;
        color: #991b1b !important;
        border: 1px solid #fecaca;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(153, 27, 27, 0.08);
    }}
    
    .thin-row {{
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.95rem;
    }}
    
    .thin-title {{
        color: #334155 !important;
        font-weight: 600;
    }}
    
    .thin-status {{
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)

# Main Banner Layout
st.markdown("""
<div class="brand-banner">
    <h1>Fake Job Posting Detection System</h1>
    <p>Analyze unstructured job postings and identify fraud using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# Load model and vectorizer assets
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "logistic_regression.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("Model assets not found. Please train models first by running the pipeline.")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_assets()

# Standard Input Form
st.markdown("#### **Enter Job Details**")
job_text = st.text_area(
    "Job Posting Content", 
    height=240,
    label_visibility="collapsed",
    placeholder="Paste job details (Title, Description, Requirements) here..."
)

if st.button("Verify Posting Authenticity"):
    if not job_text.strip():
        st.warning("Please paste some job text to perform analysis.")
    else:
        with st.spinner("Processing verification request..."):
            # 1. Clean text using NLTK lemmatizer
            cleaned = clean_text(job_text)
            
            # 2. INTENT CHECK: Dynamically verify remote status and assume verified corporate logo
            words_in_cleaned = set(cleaned.split())
            remote_keywords = {'remote', 'telecommute', 'home', 'wfh', 'online', 'virtual'}
            
            tele_val = 1 if words_in_cleaned.intersection(remote_keywords) else 0
            logo_val = 1 # Safe default baseline: assume company logo exists
            questions_val = 0
            
            # 3. Vectorize text features
            X_text_vec = vectorizer.transform([cleaned])
            
            # 4. Stack inputs
            X_num = csr_matrix([[tele_val, logo_val, questions_val]])
            X_combined = hstack([X_text_vec, X_num])
            
            # 5. Predict risk probability
            prob = model.predict_proba(X_combined)[0][1] * 100
            
            # 6. Render Thin Results Card with White Background (Glassmorphic) and 3D effects
            st.markdown('<div class="thin-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top:0; font-family:Montserrat,sans-serif;'>📄 Verification Receipt</h3>", unsafe_allow_html=True)
            
            # Output direct text in styled badges
            if prob < 50:
                st.markdown('<div class="status-badge-real">✅ REAL JOB</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-fake">❌ FAKE JOB</div>', unsafe_allow_html=True)
                
            # safety check details using looping concept
            st.markdown("<h4 style='font-family:Montserrat,sans-serif;'><b>Audit Checklist Details</b></h4>", unsafe_allow_html=True)
            
            # Define structured verification parameters
            checklist = [
                ("Linguistic Analysis Result", "Passed (Real Profile)" if prob < 50 else "High Risk Scored", "🟢" if prob < 50 else "🔴"),
                ("Self-Detected Logo Context", "Assumed Verified" if logo_val == 1 else "Unverified", "🟢"),
                ("Self-Detected Remote Context", "Yes (Work from Home)" if tele_val == 1 else "Standard Office Location", "🔵" if tele_val == 1 else "⚪"),
                ("Calculated Scam Risk Score", f"{prob:.2f}% Probability", "📈")
            ]
            
            # Looping concept over checklist entries to build grid layout
            for title, status, emoji in checklist:
                # Set dynamic text colors based on severity
                if "Risk" in status or (title == "Calculated Scam Risk Score" and prob >= 50):
                    color_code = "#991b1b" # High-contrast red
                elif "Passed" in status or "Verified" in status or (title == "Calculated Scam Risk Score" and prob < 50):
                    color_code = "#166534" # High-contrast green
                else:
                    color_code = "#5f259f" # PhonePe purple
                    
                st.markdown(f"""
                <div class="thin-row">
                    <span class="thin-title">{emoji} {title}</span>
                    <span class="thin-status" style="color: {color_code};">{status}</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
