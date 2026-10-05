import pickle
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main container background and styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card layout */
    .card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }
    
    /* Title styling */
    .title-text {
        color: #1e293b;
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle-text {
        color: #64748b;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Custom prediction banner styling */
    .placed-box {
        background-color: #d1fae5;
        border-left: 6px solid #10b981;
        padding: 20px;
        border-radius: 8px;
        color: #065f46;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    .not-placed-box {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        color: #991b1b;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the pickled scikit-learn model."""
    with open("placement.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# Header Section
st.markdown('<div class="title-text">🎓 Placement Prediction Portal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Enter your academic & resume details to predict your campus placement eligibility.</div>',
    unsafe_allow_html=True,
)

# Input Form inside a styled container
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Candidate Profile")
    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.number_input(
            "Cumulative GPA (CGPA)",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1,
            help="Enter your overall CGPA (scale of 0 - 10)",
        )

    with col2:
        resume_score = st.number_input(
            "Resume Score",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Enter your resume rating/score (scale of 0 - 10)",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Centered Action Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_btn = st.button("🚀 Predict Placement Status", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Logic & Effects
if predict_btn:
    with st.spinner("Analyzing candidate profile..."):
        # Features order must match: ['cgpa', 'resume_score']
        input_data = np.array([[cgpa, resume_score]])
        prediction = model.predict(input_data)[0]

    if prediction == 1:
        # Visual FX for Positive Prediction
        st.balloons()
        st.snow()
        st.markdown(
            """
            <div class="placed-box">
                🎉 Congratulations! You are <b>Likely to be Placed</b>!
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="not-placed-box">
                ⚠️ High Risk: <b>Unlikely to be Placed</b> based on current credentials.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary Metrics Display
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Input Summary")
    m1, m2 = st.columns(2)
    m1.metric(label="Target CGPA", value=f"{cgpa:.2f} / 10.0")
    m2.metric(label="Resume Score", value=f"{resume_score:.2f} / 10.0")
