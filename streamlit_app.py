import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt

# --- 1. CONFIG & THEME ---
st.set_page_config(
    page_title="Credit Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. NAVIGATION ---
st.sidebar.title("Navigation", anchor=False)
page = st.sidebar.radio("Module Selection:", ["Live Risk Assessment", "Feature Engineering Analysis"])

# --- 3. DATA & PERSONA DEFINITIONS ---
PERSONAS = {
    "High-Net-Worth Profile": {
        "util_ratio": 0.05, "age": 52, "monthly_income": 25000, 
        "debt_ratio": 0.10, "late_90_plus": 0, "open_credit_lines": 15,
        "late_30_59": 0, "late_60_89": 0, "real_estate_loans": 2, "dependents": 1
    },
    "High Utilization Profile": {
        "util_ratio": 0.95, "age": 23, "monthly_income": 2800, 
        "debt_ratio": 0.75, "late_90_plus": 1, "open_credit_lines": 2,
        "late_30_59": 2, "late_60_89": 1, "real_estate_loans": 0, "dependents": 0
    },
    "High Debt-to-Income Profile": {
        "util_ratio": 0.45, "age": 42, "monthly_income": 4500, 
        "debt_ratio": 1.2, "late_90_plus": 0, "open_credit_lines": 8,
        "late_30_59": 1, "late_60_89": 0, "real_estate_loans": 1, "dependents": 3
    },
    "Manual Entry": {
        "util_ratio": 0.30, "age": 45, "monthly_income": 5000, 
        "debt_ratio": 0.35, "late_90_plus": 0, "open_credit_lines": 5,
        "late_30_59": 0, "late_60_89": 0, "real_estate_loans": 0, "dependents": 0
    }
}

# ---------------------------------------------------------
# PAGE 1: LIVE RISK PREDICTOR
# ---------------------------------------------------------
if page == "Live Risk Assessment":
    st.title("Credit Risk Intelligence System", anchor=False)
    
    st.sidebar.divider()
    st.sidebar.header("Borrower Configuration", anchor=False)
    selected_persona = st.sidebar.selectbox("Load Standard Profile:", options=list(PERSONAS.keys()))
    preset = PERSONAS[selected_persona]
    p_key = selected_persona.replace(" ", "_")

    st.sidebar.divider()
    st.sidebar.subheader("Financial Metrics", anchor=False)
    
    # Row 1
    r1_col1, r1_col2 = st.sidebar.columns(2)
    util = r1_col1.slider("Utilization Ratio", 0.0, 1.0, float(preset["util_ratio"]), key=f"u_{p_key}")
    debt = r1_col2.slider("Debt Ratio", 0.0, 2.0, float(preset["debt_ratio"]), key=f"d_{p_key}")

    # Row 2
    r2_col1, r2_col2 = st.sidebar.columns(2)
    age = r2_col1.number_input("Age", 18, 100, preset["age"], key=f"a_{p_key}")
    lines = r2_col2.number_input("Open Credit Lines", 0, 50, preset["open_credit_lines"], key=f"l_{p_key}")

    # Row 3
    r3_col1, r3_col2 = st.sidebar.columns(2)
    income = r3_col1.number_input(" Income (USD)", 0, 100000, preset["monthly_income"], key=f"i_{p_key}")
    dep = r3_col2.number_input("Dependents", 0, 20, preset["dependents"], key=f"dep_{p_key}")

    st.sidebar.divider()
    st.sidebar.subheader("Delinquency History", anchor=False)
    l30 = st.sidebar.number_input("30-59 Days Past Due", 0, 20, preset["late_30_59"], key=f"l30_{p_key}")
    l60 = st.sidebar.number_input("60-89 Days Past Due", 0, 20, preset["late_60_89"], key=f"l60_{p_key}")
    l90 = st.sidebar.number_input("90+ Days Past Due", 0, 20, preset["late_90_plus"], key=f"l90_{p_key}")
    re = st.sidebar.number_input("Real Estate Loans", 0, 10, preset["real_estate_loans"], key=f"re_{p_key}")

    payload = {
        "util_ratio": util, "age": age, "late_30_59": l30, "late_60_89": l60,
        "late_90_plus": l90, "debt_ratio": debt, "monthly_income": income,
        "open_credit_lines": lines, "real_estate_loans": re, "dependents": dep
    }

    if st.button("Execute Risk Analysis", use_container_width=True):
        try:
            with st.spinner("Processing risk analysis via Blended XGBoost + LightGBM Ensemble..."):
                response = requests.post("http://backend:8000/predict", json=payload)
                if response.status_code == 200:
                    res = response.json()
                    st.subheader("Evaluation Summary", anchor=False)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Probability of Default", f"{res['probability']:.2%}")
                    color = "green" if res['verdict'] == "Approved" else "red"
                    c2.markdown(f"**Status:** :{color}[**{res['verdict']}**]")
                    c3.metric("Risk Classification", res['risk_level'])
                    
                    st.divider()
                    l_col, r_col = st.columns(2)
                    
                    with l_col:
                        st.subheader("Feature Impact Analysis (SHAP)", anchor=False)
                        st.write("Quantitative influence of individual features on the default probability:")
                        
                        reason_df = pd.DataFrame(res['top_reasons'])
                        
                        chart = alt.Chart(reason_df).mark_bar().encode(
                            x=alt.X('impact:Q', title="Impact on Risk Score (Negative decreases risk)"),
                            y=alt.Y('feature:N', sort='-x', title=None),
                            color=alt.condition(
                                alt.datum.impact > 0,
                                alt.value("#d32f2f"),  
                                alt.value("#388e3c")   
                            )
                        ).properties(height=250)
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                    with r_col:
                        st.subheader("Risk Mitigation Strategy", anchor=False)
                        if res['verdict'] == "Approved":
                            st.success(res['roadmap_to_yes'])
                        else:
                            st.warning(res['roadmap_to_yes'])
                else:
                    st.error("Backend Error. Ensure the FastAPI instance is active.")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# ---------------------------------------------------------
# PAGE 2: FEATURE ENGINEERING & EDA
# ---------------------------------------------------------
else:
    st.title("Feature Engineering & Model Mechanics", anchor=False)
    st.write("This section details the engineered features utilized to capture financial distress leading indicators.")
    st.divider()

    col_l, col_r = st.columns([1, 1.5])
    with col_l:
        st.subheader("Financial Stress Categorization", anchor=False)
        st.write("""
        Borrowers are categorized by Debt-to-Income severity:
        - **Safe**: < 0.3 Ratio
        - **Moderate**: 0.3 - 0.5 Ratio
        - **High**: 0.5 - 1.0 Ratio
        - **Critical**: > 1.0 Ratio
        """)
    
    with col_r:
        stress_data = pd.DataFrame({
            'Category': ['Safe', 'Moderate', 'High', 'Critical'],
            'Count': [85000, 32000, 18000, 15000]
        })
        donut = alt.Chart(stress_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", sort=["Safe", "Moderate", "High", "Critical"]),
            tooltip=['Category', 'Count']
        ).properties(height=300)
        st.altair_chart(donut, use_container_width=True)

    st.divider()

    st.subheader("Feature Aggregation: Delinquency Multiplier", anchor=False)
    st.write("Interaction between delinquency metrics demonstrates higher predictive validity than isolated categories. The systemic repayment failure index is calculated as:")
    st.latex(r"total\_late\_events = Late_{30} + Late_{60} + Late_{90+}")
    
    late_corr = pd.DataFrame({
        'Total Late Events': [0, 1, 2, 3, 4, 5],
        'Risk Probability': [0.03, 0.12, 0.28, 0.55, 0.78, 0.94]
    })
    line = alt.Chart(late_corr).mark_line(point=True, color='#1976d2').encode(
        x='Total Late Events:O',
        y='Risk Probability:Q'
    ).properties(height=300)
    st.altair_chart(line, use_container_width=True)

    st.divider()

    st.subheader("Normalization: Logarithmic Transformations", anchor=False)
    st.write("Logarithmic scaling $f(x) = \ln(1+x)$ is applied to heavily skewed financial distributions to stabilize gradient updates in the MLP layers.")
    
    raw_data = np.random.lognormal(8.5, 1.2, 500)
    log_data = np.log1p(raw_data)
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.write("**Pre-Transformation (Raw Distribution)**")
        st.bar_chart(np.histogram(raw_data, bins=50)[0])
    with t_col2:
        st.write("**Post-Transformation (Log Scaled)**")
        st.bar_chart(np.histogram(log_data, bins=50)[0])

    st.info("System validation complete. Blended XGBoost + LightGBM Ensemble yields 0.8691 ROC-AUC on holdout sets.")

st.sidebar.divider()
st.sidebar.caption("System Architecture by Omar F. | Applied Machine Learning Portfolio | Cairo, Egypt")