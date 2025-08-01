import streamlit as st

# home page(landing page)
home = st.Page("pages/Home.py", icon='🏠')  # Main landing page

# model 1
m1 = st.Page("pages/Face_Authenticity_Checker.py", icon='🔬')  
m1_eda = st.Page("pages/Face_Authenticity_EDA.py", icon='🔁') 

# model 2
m2 = st.Page("pages/Image_Authenticity_Checker.py", icon='🔬')  
m2_eda = st.Page("pages/Image_Authenticity_EDA.py", icon='🔁') 

# model 3
m3 = st.Page("pages/Signature_Verification.py", icon='🔬')  
m3_eda = st.Page("pages/Signature_Verification_EDA.py", icon='🔁') 

# project info
credits = st.Page("pages/Credits.py", icon='🧠') 
tech = st.Page("pages/Tech_Stack.py", icon='🛠️') 
reviews = st.Page("pages/Reviews.py", icon='📨') 

# pg = st.navigation(
#     [home,data_analysis,sythentic,credits,tech,reviews]
# )

pg = st.navigation({
    "Home": [home],
    "Face Classifier": [m1,m1_eda],
    "Image Authenticator": [m2,m2_eda],
    "Signature Verifier": [m3,m3_eda],
    "Project Info": [credits,tech,reviews],
})

pg.run()
