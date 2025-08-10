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
    color: #4B5563 !important;     
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

## -------------------------------
# Title Section
# -------------------------------
st.markdown('<h1 class="custom-h1">Credits</h1>', unsafe_allow_html=True)

# -------------------------------
# Author Block
# -------------------------------
with st.container(border=True):
    st.markdown("""
    <div class="module-box">
        <h2 style="margin-bottom: 0.5rem;">Made By</h2>
        <h3 style="margin-top: 0;">Shail K Patel</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.link_button("LinkedIn Profile", "https://www.linkedin.com/in/shail-k-patel/", icon="🔗", use_container_width=True)
    st.link_button("GitHub Profile", "https://github.com/ShailKPatel", icon="🐙", use_container_width=True)
    st.link_button("Portfolio Website", "https://shailkpatel.github.io/", icon="🌐", use_container_width=True)

# -------------------------------
# Repository Block
# -------------------------------
with st.container(border=True):
    st.markdown("""
    <div class="module-box">
        <h2 style="margin-bottom: 0.5rem;">GitHub Repository</h2>
        <div style="align-items: center;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #2563eb;
                color: white;
                padding: 0.6rem 1.2rem;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                margin-top: 0.5rem;
            ">
                🔗 Visit Repository
            </button>
        </a>
        </div>
    </div>
    """, unsafe_allow_html=True)



# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="background-color: #111827; padding: 10px 0; max-width: 100%; margin-bottom: 0;">
    <div style="text-align: center; color: gray;">
        © 2025 Shail K Patel · Crafted out of boredom.
    </div>
    <div style="text-align: center; color: gray;">
        <a href="https://github.com/ShailKPatel/Falsifi.AI/" style="color: gray; text-decoration: none;">GitHub Repo</a> · MIT License
    </div>
</div>
""", unsafe_allow_html=True)

