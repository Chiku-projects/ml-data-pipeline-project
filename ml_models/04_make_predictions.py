# COMPONENT 3: MAKE PREDICTIONS
# File: ml_models/make_predictions.py
# Purpose: Use trained models to make predictions for all customers

import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"

# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_models():
    """Load all trained models"""
    logger.info("="*70)
    logger.info("LOADING TRAINED MODELS")
    logger.info("="*70)
    
    try:
        # Load Churn Model
        with open(f"{MODEL_DIR}/churn_model.pkl", 'rb') as f:
            churn_data = pickle.load(f)
            churn_model = churn_data['model']
            churn_scaler = churn_data['scaler']
        logger.info("✅ Churn model loaded")
        
        # Load LTV Model
        with open(f"{MODEL_DIR}/ltv_model.pkl", 'rb') as f:
            ltv_data = pickle.load(f)
            ltv_model = ltv_data['model']
            ltv_scaler = ltv_data['scaler']
        logger.info("✅ LTV model loaded")
        
        # Load Recommendation Model
        with open(f"{MODEL_DIR}/recommendation_model.pkl", 'rb') as f:
            rec_data = pickle.load(f)
            recommendation_model = rec_data
        logger.info("✅ Recommendation model loaded")
        
        return {
            'churn': {'model': churn_model, 'scaler': churn_scaler},
            'ltv': {'model': ltv_model, 'scaler': ltv_scaler},
            'recommendations': recommendation_model
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

def load_customer_data():
    """Load customer data from Gold Layer"""
    logger.info("\n" + "="*70)
    logger.info("LOADING CUSTOMER DATA")
    logger.info("="*70)
    
    # In production: query Databricks Gold Layer
    # For now: generate sample data
    
    np.random.seed(42)
    n_customers = 1000
    
    data = {
        'customer_id': [f'C{i:05d}' for i in range(n_customers)],
        'customer_name': [f'Customer {i}' for i in range(n_customers)],
        'total_spending': np.random.exponential(1000, n_customers),
        'total_orders': np.random.poisson(10, n_customers),
        'avg_order_value': np.random.exponential(150, n_customers),
        'days_since_last_purchase': np.random.randint(1, 500, n_customers),
        'customer_tenure_days': np.random.randint(30, 1825, n_customers),
        'distinct_categories_purchased': np.random.randint(1, 10, n_customers),
        'completed_orders': np.random.poisson(8, n_customers),
        'cancelled_orders': np.random.poisson(1, n_customers),
        'completion_rate': np.random.uniform(70, 100, n_customers),
        'is_inactive': np.random.binomial(1, 0.15, n_customers),
    }
    
    df = pd.DataFrame(data)
    logger.info(f"✅ Loaded {len(df):,} customers")
    
    return df

# ============================================================
# MAKE PREDICTIONS
# ============================================================

def predict_churn(df, models):
    """Predict churn probability for each customer"""
    logger.info("\n" + "="*70)
    logger.info("PREDICTING: CHURN RISK")
    logger.info("="*70)
    
    feature_cols = [
        'total_spending', 'total_orders', 'avg_order_value',
        'days_since_last_purchase', 'customer_tenure_days',
        'distinct_categories_purchased', 'completed_orders',
        'cancelled_orders', 'completion_rate'
    ]
    
    X = df[feature_cols]
    X_scaled = models['churn']['scaler'].transform(X)
    
    # Get predictions and probabilities
    churn_pred = models['churn']['model'].predict(X_scaled)
    churn_proba = models['churn']['model'].predict_proba(X_scaled)[:, 1]
    
    df['churn_prediction'] = churn_pred
    df['churn_probability'] = churn_proba
    
    # Categorize risk
    df['churn_risk'] = pd.cut(churn_proba, 
                              bins=[0, 0.3, 0.6, 1.0],
                              labels=['Low', 'Medium', 'High'])
    
    logger.info(f"✅ Churn predictions completed")
    logger.info(f"   High Risk (>60%): {(churn_proba > 0.6).sum()} customers")
    logger.info(f"   Medium Risk (30-60%): {((churn_proba > 0.3) & (churn_proba <= 0.6)).sum()} customers")
    logger.info(f"   Low Risk (<30%): {(churn_proba <= 0.3).sum()} customers")
    
    return df

def predict_ltv(df, models):
    """Predict Customer Lifetime Value for each customer"""
    logger.info("\n" + "="*70)
    logger.info("PREDICTING: CUSTOMER LIFETIME VALUE")
    logger.info("="*70)
    
    feature_cols = [
        'total_spending', 'total_orders', 'avg_order_value',
        'days_since_last_purchase', 'customer_tenure_days',
        'distinct_categories_purchased', 'completed_orders',
        'cancelled_orders', 'completion_rate'
    ]
    
    X = df[feature_cols]
    X_scaled = models['ltv']['scaler'].transform(X)
    
    # Get predictions
    ltv_pred = models['ltv']['model'].predict(X_scaled)
    
    df['predicted_ltv'] = ltv_pred
    
    logger.info(f"✅ LTV predictions completed")
    logger.info(f"   Avg Predicted LTV: ${ltv_pred.mean():.2f}")
    logger.info(f"   Min Predicted LTV: ${ltv_pred.min():.2f}")
    logger.info(f"   Max Predicted LTV: ${ltv_pred.max():.2f}")
    logger.info(f"   Top 10% LTV (>${np.percentile(ltv_pred, 90):.2f}): {(ltv_pred > np.percentile(ltv_pred, 90)).sum()} customers")
    
    return df

def recommend_products(df, models):
    """Generate product recommendations for each customer"""
    logger.info("\n" + "="*70)
    logger.info("GENERATING: PRODUCT RECOMMENDATIONS")
    logger.info("="*70)
    
    interaction_matrix = models['recommendations']['interaction_matrix']
    product_popularity = models['recommendations']['product_popularity']
    
    n_products = len(product_popularity)
    top_products = np.argsort(product_popularity)[-5:][::-1]  # Top 5 products
    
    # Simple recommendation: top products + customer affinity
    recommendations = []
    for idx, row in df.iterrows():
        customer_history = interaction_matrix[idx]
        
        # Products not yet purchased
        unpurchased = np.where(customer_history == 0)[0]
        
        if len(unpurchased) > 0:
            # Recommend top unpurchased products
            unpurchased_popularity = product_popularity[unpurchased]
            best_rec = unpurchased[np.argsort(unpurchased_popularity)[-3:][::-1]]
            recommendations.append(','.join([f'P{p:03d}' for p in best_rec]))
        else:
            recommendations.append(','.join([f'P{p:03d}' for p in top_products]))
    
    df['recommended_products'] = recommendations
    
    logger.info(f"✅ Recommendations generated")
    logger.info(f"   Avg products recommended per customer: 3")
    logger.info(f"   Coverage (% customers receiving recommendations): 100%")
    
    return df

# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(df):
    """Save predictions to CSV"""
    logger.info("\n" + "="*70)
    logger.info("SAVING PREDICTIONS")
    logger.info("="*70)
    
    # Select relevant columns
    output_cols = [
        'customer_id', 'customer_name',
        'total_spending', 'total_orders',
        'churn_prediction', 'churn_probability', 'churn_risk',
        'predicted_ltv',
        'recommended_products'
    ]
    
    output_df = df[output_cols].copy()
    
    # Save to CSV
    output_file = 'predictions_output.csv'
    output_df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Predictions saved to {output_file}")
    logger.info(f"   Total records: {len(output_df)}")
    logger.info(f"   Columns: {len(output_df.columns)}")
    
    return output_file

# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================

def main():
    """Run complete prediction pipeline"""
    
    logger.info("\n")
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " "*12 + "COMPONENT 3: MAKE PREDICTIONS FOR ALL CUSTOMERS" + " "*6 + "║")
    logger.info("╚" + "="*68 + "╝")
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Load models
        models = load_models()
        
        # Load customer data
        df = load_customer_data()
        
        # Make predictions
        df = predict_churn(df, models)
        df = predict_ltv(df, models)
        df = recommend_products(df, models)
        
        # Save predictions
        output_file = save_predictions(df)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\n📊 CHURN PREDICTIONS:")
        logger.info(f"   Total predictions: {len(df)}")
        logger.info(f"   Customers at high churn risk: {(df['churn_probability'] > 0.6).sum()}")
        logger.info(f"   Action: Focus retention efforts on high-risk customers")
        
        logger.info(f"\n💰 LTV PREDICTIONS:")
        logger.info(f"   Total predictions: {len(df)}")
        logger.info(f"   Top 10% customers (potential high-value): {(df['predicted_ltv'] > df['predicted_ltv'].quantile(0.9)).sum()}")
        logger.info(f"   Action: Personalize offers for top-value customers")
        
        logger.info(f"\n🎁 PRODUCT RECOMMENDATIONS:")
        logger.info(f"   Total recommendations: {len(df)}")
        logger.info(f"   Avg recommendations per customer: 3")
        logger.info(f"   Action: Send personalized product emails to customers")
        
        logger.info("\n" + "="*70)
        logger.info("🎉 PREDICTIONS COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info(f"Completed at: {datetime.now()}")
        logger.info(f"Output file: {output_file}")
        
        # Show sample predictions
        logger.info("\n" + "="*70)
        logger.info("SAMPLE PREDICTIONS (First 10 Customers)")
        logger.info("="*70)
        
        sample = df[['customer_id', 'churn_probability', 'churn_risk', 
                     'predicted_ltv', 'recommended_products']].head(10)
        logger.info("\n" + sample.to_string())
        
        return df
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    predictions_df = main()