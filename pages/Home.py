import streamlit as st

st.set_page_config(page_title="Falsifi.AI", layout="wide")


# -------------------------------
# Inject Global Dark Theme + Custom Text Selection
# -------------------------------
st.markdown("""
<style>
/* Global dark background and white text */
body {
    background-color: black !important;
    color: white !important;
}

/* All text elements white (except h1 which is overridden by class) */
h2, h3, h4, h5, h6, p, li, a, label {
    color: white;
}

/* Custom H1 color override */
.custom-h1 {
    color: #374151 !important;     /* Dark gray */
    font-weight: 800 !important;   /* Bold */
    font-size: 3.5rem !important;  /* Larger font size */
}


/* Custom text selection style */
::selection {
    background: white;
    color: black;
}
::-moz-selection {
    background: white;
    color: black;
}

/* Clean block spacing */
.block-container {
    padding: 2rem 1rem;
}

/* Module boxes */
.module-box {
    background-color: #111;
    padding: 1.5rem;
    border: 1px solid #444;
    border-radius: 10px;
    margin-bottom: 1rem;
    height: 100%;
}
.module-box a {
    color: white !important;
    font-weight: bold;
    text-decoration: none;
}

.custom-h1 {
    color: #374151 !important;     
    font-weight: 800 !important;   
    font-size: 3.75rem !important;  
    margin-bottom: 0rem !important;
    margin-top: 0rem !important;
}

.custom-subtitle {
    font-size: 1.25rem !important;
    color: white !important;
    margin-top: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section 
# -------------------------------
st.markdown("""
<h1 class="custom-h1">Falsifi.AI</h1>
<p class="custom-subtitle">Deep Learning-Based Forgery Detection for Faces, Images, and Signatures</p>
""", unsafe_allow_html=True)

# -------------------------------
# Introduction Box
# -------------------------------
st.markdown("""
<div class="module-box">
    <h3>What is Falsifi.AI?</h3>
    <p>Falsifi.AI is a deep-learning powered toolkit to detect:</p>
    <ul>
        <li>AI-generated human faces (GAN detection)</li>
        <li>Manipulated images (cloning, erasure, splicing)</li>
        <li>Forged handwritten signatures</li>
    </ul>
    <p>Made for journalists, forensic professionals, and digital identity teams.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Modules Section
# -------------------------------
st.markdown("## Models")

cols = st.columns(3)

with cols[0]:
    with st.container(border=True):
        st.markdown("""
        <div class="module-box">
            <h4>🔬 Fake Face Detector</h4>
            <p>This model detects whether a given face image is real or AI-generated.</p>
            <p>It’s trained on diverse datasets of authentic and synthetic faces. Useful for verifying profile images in KYC, recruitment, and social media platforms.</p>
            <p><b>Input:</b> Face photo · <b>Output:</b> Real / Fake + Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/Face_Authenticity_Checker.py", label="Launch →", icon="🔬")

with cols[1]:
    with st.container(border=True):
        st.markdown("""
        <div class="module-box">
            <h4>🖼️ Image Manipulation Detector</h4>
            <p>This model detects whether an image has been edited and highlights the regions using heatmaps for visual inspection.</p>     
            <p>Ideal for verifying authenticity of photos, certificates, legal evidence, or ID documents shared online.</p>
            <p><b>Input:</b> General image · <b>Output:</b> Tampered / Untampered + Visual Mask</p>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/Image_Authenticity_Checker.py", label="Launch →", icon="🖼️")

with cols[2]:
    with st.container(border=True):
        st.markdown("""
        <div class="module-box">
            <h4>✍️ Signature Forgery Detector</h4>
            <p>This model evaluates pairs of handwritten signatures to detect forgeries. It uses datasets CEDAR and GPDS.</p>
            <p>Applicable for banking contracts, academic certificates, or any scenario requiring written consent verification.</p>
            <p><b>Input:</b> Two signatures · <b>Output:</b> Match / Mismatch + Similarity Score</p>
        </div>
        """, unsafe_allow_html=True)
        st.page_link("pages/Signature_Verification.py", label="Launch →", icon="✍️")


# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="background-color: #111827; padding: 20px 0; max-width: 100%; margin-bottom: 0;">
    <div style="text-align: center; color: gray;">
        © 2025 Shail K Patel · Crafted out of boredom.
    </div>
    <div style="text-align: center; color: gray;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI/" style="color: gray; text-decoration: none;">GitHub Repo</a> · MIT License
    </div>
</div>
""", unsafe_allow_html=True)
