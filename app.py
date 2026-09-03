import streamlit as st
import pandas as pd
import importlib.util
import base64

# ---------------------------------------
# PAGE CONFIG (must be first)
# ---------------------------------------
st.set_page_config(
    page_title="Patient Risk Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# CUSTOM CSS for premium, clean look
# ---------------------------------------
st.markdown("""
<style>
    /* main background */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* card-like containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.15s ease;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    /* sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: #ffffff;
        border-right: 1px solid rgba(0,0,0,0.04);
    }
    
    /* headers */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
        font-weight: 500 !important;
        letter-spacing: -0.01em;
        color: #1a2a3a;
    }
    
    /* buttons */
    .stButton button {
        background: #1a3a5c;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 2rem;
        font-weight: 450;
        letter-spacing: 0.2px;
        transition: 0.15s;
    }
    .stButton button:hover {
        background: #0f2a44;
        box-shadow: 0 2px 8px rgba(20,60,120,0.12);
    }
    
    /* dataframes */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.04);
    }
    
    /* upload box */
    .stFileUploader > div {
        border: 2px dashed #d0d8e0;
        border-radius: 16px;
        padding: 2rem 1rem;
        background: #fafcfe;
        transition: 0.15s;
    }
    .stFileUploader > div:hover {
        border-color: #1a3a5c;
        background: #f6f9ff;
    }
    
    /* divider */
    hr {
        margin: 1.5rem 0;
        opacity: 0.3;
    }
    
    /* footer */
    .footer {
        text-align: center;
        color: #7a8a9a;
        font-size: 0.78rem;
        padding: 1.5rem 0 0.2rem;
        border-top: 1px solid rgba(0,0,0,0.04);
        margin-top: 2rem;
    }
    
    /* success/warning boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
    }
    
    /* captions */
    .caption {
        color: #6a7a8a;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# LOAD 08_predict.py
# ---------------------------------------
spec = importlib.util.spec_from_file_location("predict_module", "08_predict.py")
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)

# ---------------------------------------
# HEADER
# ---------------------------------------
st.markdown("""
<div style="margin-bottom: 1.8rem;">
    <h1 style="font-size: 2.4rem; margin: 0; color: #0b1e2e; font-weight: 500;">
        Patient Risk Analysis
    </h1>
    <p style="font-size: 1rem; color: #4a5a6a; margin: 0.2rem 0 0.5rem 0;">
        Machine learning prediction for Diabetes, Hypertension, and Heart Disease
    </p>
    <div style="height: 3px; width: 80px; background: #1a3a5c; border-radius: 2px; margin: 0.3rem 0 0.2rem 0;"></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------
with st.sidebar:
    st.markdown("### Prediction Settings")
    st.markdown("---")
    target = st.selectbox(
        "Select Disease",
        ["Diabetes", "Hypertension", "Heart_Disease"],
        help="Choose the condition to evaluate patient risk."
    )
    st.markdown("---")
    st.caption("MEPS 2020 dataset")

# ---------------------------------------
# MAIN AREA
# ---------------------------------------
uploaded_file = st.file_uploader(
    "Upload Patient Data (CSV)",
    type=["csv"],
    help="Upload a CSV file containing patient records with clinical features."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df)} patient records.")

        # Data preview
        col1, col2 = st.columns([2.2, 1])
        with col1:
            st.markdown("#### Data Preview")
            st.dataframe(df.head(5), use_container_width=True, height=220)
        with col2:
            st.markdown("#### Summary")
            st.markdown(f"""
            <div style="background: #f7f9fc; border-radius: 10px; padding: 0.8rem 1.2rem; border: 1px solid rgba(0,0,0,0.03);">
                <b>Rows</b>  {df.shape[0]}<br>
                <b>Columns</b>  {df.shape[1]}<br>
                <b>Missing</b>  {df.isnull().sum().sum()}
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if st.button("Predict Patient Risk", type="primary"):
            with st.spinner("Analyzing patient data..."):
                result = predict_module.predict(df, target)

            st.success("Prediction completed successfully.")

            # Metrics
            total = len(result)
            flagged = result["Flagged_For_Screening"].sum()
            avg_risk = result["Risk_Probability"].mean()
            high_risk = result["Risk_Tier"].astype(str).eq("High").sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Patients", total)
            col2.metric("Flagged for Screening", flagged)
            col3.metric("High Risk Patients", high_risk)
            col4.metric("Average Risk", f"{avg_risk:.2%}")

            st.divider()

            # Results table
            st.markdown("#### Prediction Results")
            st.dataframe(result, use_container_width=True, height=320)

            # Charts
            col_left, col_right = st.columns(2, gap="medium")
            with col_left:
                st.markdown("#### Risk Distribution")
                risk_counts = result["Risk_Tier"].value_counts().reset_index()
                risk_counts.columns = ["Risk Tier", "Patients"]
                st.bar_chart(risk_counts.set_index("Risk Tier"), height=240)

            with col_right:
                st.markdown("#### Highest Risk Patients")
                top = result.nlargest(min(8, len(result)), "Risk_Probability")
                st.dataframe(
                    top[["Risk_Probability", "Risk_Tier", "Flagged_For_Screening"]],
                    use_container_width=True,
                    height=240
                )

            st.divider()

            # Download
            csv = result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'''
            <a href="data:file/csv;base64,{b64}" 
               download="{target.lower()}_predictions.csv" 
               style="background: #1a3a5c; color: white; padding: 0.5rem 1.8rem; 
                      border-radius: 8px; text-decoration: none; font-weight: 450;
                      display: inline-block; transition: 0.15s;">
                Download Predictions CSV
            </a>
            '''
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error("Prediction failed. Please check the file format.")
        st.exception(e)

else:
    # Placeholder
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; 
                justify-content: center; padding: 3rem 1rem; 
                background: #f7f9fc; border-radius: 20px; 
                border: 1px solid #e8edf2; margin: 0.5rem 0;">
        <span style="font-size: 3.2rem; opacity: 0.5; margin-bottom: 0.5rem;">📊</span>
        <h3 style="color: #1a2a3a; font-weight: 450; margin: 0.2rem 0;">Upload CSV to Begin</h3>
        <p style="color: #5a6a7a; font-size: 0.95rem;">Upload a patient data file to start risk prediction</p>
        <div style="background: #eef2f7; padding: 0.2rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #4a5a6a; margin-top: 0.4rem;">
            CSV format
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("""
<div class="footer">
    Patient Risk Analysis · MEPS 2020 · Machine Learning Prediction System
</div>
""", unsafe_allow_html=True)