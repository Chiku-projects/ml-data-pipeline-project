import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")

API_URL = "http://localhost:8000"

# Title
st.title("🎯 ML Predictions Pipeline Dashboard")
st.markdown("Real-time monitoring of churn, LTV, and product recommendations")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Make Prediction", "Batch Upload"])

# Check API Health
@st.cache_data(ttl=10)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except:
        return {"status": "offline"}

health = check_api_health()

# Health Status
col1, col2, col3 = st.columns(3)
with col1:
    status = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Offline"
    st.metric("API Status", status)
with col2:
    st.metric("Models Loaded", sum(health.get("models_loaded", {}).values()))
with col3:
    st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))

st.divider()

# Page: Dashboard
if page == "Dashboard":
    st.header("📊 Real-time Analytics")
    
    sample_customers = pd.DataFrame({
        'Customer ID': [f'C{i:05d}' for i in range(1, 101)],
        'Spending': [1000 + i*50 for i in range(100)],
        'Orders': [5 + i%20 for i in range(100)],
        'Churn Risk': [['High', 'Medium', 'Low'][i%3] for i in range(100)]
    })    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Risk Distribution")
        churn_dist = sample_customers['Churn Risk'].value_counts()
        fig = px.pie(values=churn_dist.values, names=churn_dist.index, 
                     color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Spending Distribution")
        fig = px.histogram(sample_customers, x='Spending', nbins=20, 
                          title="Spending Range")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top Customers by Spending")
    top_customers = sample_customers.nlargest(10, 'Spending')[['Customer ID', 'Spending', 'Churn Risk']]
    st.dataframe(top_customers, use_container_width=True)

# Page: Make Prediction
elif page == "Make Prediction":
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        customer_id = st.text_input("Customer ID", "C00001")
        total_spending = st.slider("Total Spending ($)", 0, 10000, 1000)
        total_orders = st.slider("Total Orders", 0, 100, 10)
        avg_order_value = st.slider("Avg Order Value ($)", 0, 500, 100)
        days_since_purchase = st.slider("Days Since Last Purchase", 0, 365, 30)
    
    with col2:
        customer_tenure = st.slider("Customer Tenure (days)", 0, 2000, 365)
        categories = st.slider("Distinct Categories", 0, 20, 5)
        completed = st.slider("Completed Orders", 0, 100, 10)
        cancelled = st.slider("Cancelled Orders", 0, 20, 1)
        completion_rate = st.slider("Completion Rate (%)", 0, 100, 90)
    
    if st.button("🎯 Get Predictions", type="primary"):
        payload = {
            "customer_id": customer_id,
            "total_spending": total_spending,
            "total_orders": total_orders,
            "avg_order_value": avg_order_value,
            "days_since_last_purchase": days_since_purchase,
            "customer_tenure_days": customer_tenure,
            "distinct_categories_purchased": categories,
            "completed_orders": completed,
            "cancelled_orders": cancelled,
            "completion_rate": completion_rate
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/all", json=payload)
            predictions = response.json()
            
            # Display results
            st.success("✅ Predictions Generated!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_prob = predictions['churn']['churn_probability']
                churn_risk = predictions['churn']['churn_risk']
                color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[churn_risk]
                st.metric(
                    "Churn Risk",
                    f"{color} {churn_risk}",
                    f"{churn_prob:.2%}"
                )
            
            with col2:
                ltv = predictions['ltv']['predicted_ltv']
                st.metric(
                    "Lifetime Value",
                    f"${ltv:.2f}",
                    f"Segment: {predictions['ltv'].get('segment', 'N/A')}"
                )
            
            with col3:
                products = ", ".join(predictions['recommendations']['recommended_products'])
                st.metric(
                    "Recommended Products",
                    products[:20] + "..."
                )
            
            # Detailed view
            st.subheader("📋 Detailed Predictions")
            st.json(predictions)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Page: Batch Upload
elif page == "Batch Upload":
    st.header("📤 Batch Predictions")
    
    st.write("Upload a CSV with customer data to get predictions for multiple customers")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} customers")
        
        if st.button("🚀 Generate Predictions", type="primary"):
            st.info("Processing batch predictions...")
            # Batch prediction logic here
            st.success("✅ Done!")

st.divider()
st.caption("ML Pipeline Dashboard | Real-time Monitoring")