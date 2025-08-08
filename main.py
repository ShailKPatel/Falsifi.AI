import streamlit as st

# home page(landing page)
home = st.Page("pages/Home.py", icon='🏠')  # Main landing page

# signature verification pages
m3 = st.Page("pages/Signature_Verification.py", icon='🔬')  
m3_eda = st.Page("pages/Signature_Verification_EDA.py", icon='🔁') 

# dataset overview page
data_analysis = st.Page("pages/Dataset_Overview.py", icon='📊')

# project info
credits = st.Page("pages/Credits.py", icon='🧠') 
tech = st.Page("pages/Tech_Stack.py", icon='🛠️') 
reviews = st.Page("pages/Reviews.py", icon='📨') 

# pg = st.navigation(
#     [home,data_analysis,sythentic,credits,tech,reviews]
# )

pg = st.navigation({
    "Home": [home],
    "Signature Verifier": [m3,m3_eda,data_analysis],
    "Project Info": [credits,tech,reviews],
})

pg.run()
