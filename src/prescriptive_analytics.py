import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
@st.cache_resource
def load_ml_assets():
    """Loads Member 2's saved model and extracts the pipeline."""
    # Load the dictionary saved by churn_model.py
    saved_assets = joblib.load("models/churn_model.joblib")
    pipeline = saved_assets["model"]
    threshold = saved_assets["threshold"]
    feature_names = saved_assets["features"]
    return pipeline, threshold, feature_names

# ==========================================
# 2. DIAGNOSTIC LAYER: SHAP EXPLAINER
# ==========================================
def render_shap_explanation(pipeline, feature_names, customer_data):
    """Generates a SHAP waterfall plot to explain WHY a customer is leaving."""
    st.subheader("Diagnostic Analytics: Root-Cause Churn Analysis")
    st.write("Using Game Theory (SHAP) to isolate the exact business drivers causing this customer's churn risk.")
    
    # Extract the classifier and preprocessor from Member 2's pipeline
    preprocessor = pipeline.named_steps["prep"]
    model = pipeline.named_steps["clf"]
    
    # Preprocess the specific customer's data
    processed_data = preprocessor.transform(customer_data[feature_names])
    
    # Generate SHAP values safely using the generic Explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(processed_data)
    
    # Plot the explanation for the dashboard
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close()

# ==========================================
# 3. PRESCRIPTIVE LAYER: WHAT-IF SIMULATOR
# ==========================================
def render_what_if_simulator(pipeline, customer_data, base_ltv=1200):
    """Calculates Net Profit ROI based on targeted retention discounts."""
    st.subheader("Prescriptive Analytics: Profit Optimization Simulator")
    st.write("Adjust the proposed discount below to simulate the financial ROI of retaining this customer.")
    
    # Get the baseline prediction (before any intervention)
    baseline_prob = pipeline.predict_proba(customer_data)[:, 1][0]
    
    # The Streamlit Interactive Slider
    proposed_discount = st.slider("Select Targeted Discount (%)", min_value=0, max_value=30, value=10, step=5)
    
    # Simulate the data change (Assuming your dataset has a 'monthly_charges' or similar pricing column)
    # UPDATE 'monthly_charges' to exactly match Member 1's dataset column name
    simulated_data = customer_data.copy()
    if 'monthly_charges' in simulated_data.columns:
        discount_multiplier = (100 - proposed_discount) / 100
        simulated_data['monthly_charges'] = simulated_data['monthly_charges'] * discount_multiplier
        
    # Get the new prediction based on the lowered price
    new_prob = pipeline.predict_proba(simulated_data)[:, 1][0]
    
    # Calculate Business ROI Math
    risk_reduction = baseline_prob - new_prob
    cost_of_discount = base_ltv * (proposed_discount / 100)
    expected_value_saved = base_ltv * risk_reduction
    net_profit = expected_value_saved - cost_of_discount
    
    # Display the Financial Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Churn Risk", f"{baseline_prob:.1%}")
    col2.metric("New Simulated Risk", f"{new_prob:.1%}", delta=f"{-risk_reduction:.1%}", delta_color="inverse")
    
    # Conditional formatting for the Profit Output
    if net_profit > 0:
        col3.metric("Net Profit Saved", f"${net_profit:.2f}", delta="Profitable Intervention")
        st.success(f"Recommendation: APPROVE. Offering a {proposed_discount}% discount mathematically maximizes net revenue.")
    else:
        col3.metric("Net Profit Saved", f"${net_profit:.2f}", delta="Loss-Making", delta_color="inverse")
        st.error(f"Recommendation: REJECT. The cost of the discount outweighs the statistical value of retaining this user.")

# ==========================================
# 4. MAIN EXECUTION BLOCK (For app.py)
# ==========================================
# You will plug this into the main app.py file where Member 1 left off.
if __name__ == "__main__":
    st.title("Smart Customer Retention System")
    
    pipeline, threshold, feature_names = load_ml_assets()
    
    # For demonstration, grab a single high-risk customer from the dataset
    # df = pd.read_csv("data/processed/clean_data.csv")
    # customer_data = df.iloc[[0]] 
    
    # render_shap_explanation(pipeline, feature_names, customer_data)
    # render_what_if_simulator(pipeline, customer_data)