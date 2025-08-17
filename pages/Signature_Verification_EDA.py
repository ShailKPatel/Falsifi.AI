import streamlit as st
import pandas as pd
import os


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

.download-btn-link {
    display: inline-block;
    padding: 0.5rem 1rem;
    background-color: #1a73e8;
    color: white !important;
    text-decoration: none;
    border-radius: 5px;
    font-weight: bold;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.download-btn-link:hover {
    background-color: #1669d2;
    color: white !important;
}

/* Tooltip container */
.tooltip {
  position: relative;
  display: inline-block;
  cursor: pointer;
  border-bottom: 1px dotted white;
}

/* Tooltip text */
.tooltip .tooltiptext {
  visibility: hidden;
  width: 300px;
  background-color: #333;
  color: #fff;
  text-align: left;
  border-radius: 6px;
  padding: 8px;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  margin-left: -150px;
  opacity: 0;
  transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section 
# -------------------------------
st.markdown("""
<h1 class="custom-h1">Signature Verification EDA</h1>
""", unsafe_allow_html=True)

# -------------------------------
# Introduction
# -------------------------------
st.markdown("""
<div class="module-box">
    <h4>About This Page</h4>
    <p>This page presents a comprehensive overview and performance results for the various deep learning models and pipelines tested in this research project. Each model has been trained and evaluated for the task of handwritten signature verification. Below, you'll find a detailed breakdown for each model, including its architecture, performance metrics, and a link to download the trained model weights. You can also view the complete training script for each model.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Model Results
# -------------------------------
st.markdown("---")

try:
    # Construct absolute path to the CSV file
    # The script is in 'pages', so we go up one level to the project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_path = os.path.join(project_root, 'models', 'model_result_log.csv')
    
    if not os.path.exists(results_path):
        st.error(f"Model results file not found. Looked for it at: `{results_path}`")
    else:
        df = pd.read_csv(results_path)

        for index, row in df.iterrows():
            model_id = row['id']
            
            st.markdown(f"### Model {model_id}: {row['Model_Name']}")
            
            with st.container():
                st.markdown(f"*{row['Description']}*")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("<h5><b>Model & Pipeline Details</b></h5>", unsafe_allow_html=True)
                    
                    if 'Justification' in row and pd.notna(row['Justification']):
                        justification_html = row['Justification'].replace("'", "&apos;").replace('"', '&quot;')
                        st.markdown(f"""
                        <div class="tooltip">Why was this model chosen?
                          <span class="tooltiptext">{justification_html}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"**Pipeline Type:** `{row['Pipeline']}`")
                    st.markdown(f"**Feature Extractor:** `{row['Feature_Extractor']}`")
                    st.markdown(f"**Loss Function:** `{row['Loss']}`")

                with col2:
                    st.markdown("<h5><b>Performance Metrics</b></h5>", unsafe_allow_html=True)
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'F1-Score', 'Recall', 'Precision', 'Best Loss'],
                        'Value': [row['Accuracy'], row['F1'], row['Recall'], row['Precision'], row['Best_Loss']]
                    })
                    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
                    st.table(metrics_df.set_index('Metric'))

                # Download button and code expander
                download_url = f"https://huggingface.co/ShailKPatel/Signature_Verification_{model_id}/resolve/main/model_{model_id}.pth"
                st.markdown(f'<a href="{download_url}" class="download-btn-link" target="_blank">Download Model Weights (Model {model_id})</a>', unsafe_allow_html=True)

                model_code_path = os.path.join(project_root, 'models', f'model_{model_id}.py')
                if os.path.exists(model_code_path):
                    with open(model_code_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    with st.expander(f"View Training Code for Model {model_id}"):
                        st.code(code, language='python')
                else:
                    st.warning(f"Training code for Model {model_id} not found.")
            
            st.markdown("---") # Separator for next model

except FileNotFoundError:
    st.error("Could not find the model results log file: `models/model_result_log.csv`")
except Exception as e:
    st.error(f"An error occurred while displaying model results: {e}")


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