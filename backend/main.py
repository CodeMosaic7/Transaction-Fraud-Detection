import streamlit as st
import pandas as pd
from model.prediction_model import preprocess_and_split_data, train_random_forest, evaluate_model, train_xgboost, load_data
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Fraud Detection Model",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-result {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
    }
    .safe-alert {
        background-color: #44ff44;
        color: white;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def save_model(model, filename):
    """Save trained model to file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load trained model from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def train_model():
    """Train the fraud detection model"""
    with st.spinner("Training model... This may take a few minutes."):
        try:
            # Load and preprocess data
            data = load_data('data/data.csv')
            X_train, X_test, y_train, y_test = preprocess_and_split_data(data, target_col='is_fraud')
            print(X_train.info())

            # Train Random Forest model
            rf_model = train_random_forest(X_train, y_train)

            # Evaluate model
            rf_metrics = evaluate_model(rf_model, X_test, y_test)

            # Save model
            save_model(rf_model, 'trained_model.pkl')

            # ✅ Save feature names used in training
            with open("model_columns.pkl", "wb") as f:
                pickle.dump(X_train.columns.tolist(), f)

            # Store in session state
            st.session_state.model = rf_model
            st.session_state.metrics = rf_metrics
            st.session_state.feature_names = X_train.columns.tolist()

            st.success("✅ Model trained successfully!")

            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{rf_metrics['accuracy']}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{rf_metrics['precision']}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{rf_metrics['recall']}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <h2>{rf_metrics['f1_score']}</h2>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")

def predict_fraud(input_data):
    """Make fraud prediction"""
    model = st.session_state.get('model') or load_model('trained_model.pkl')
    
    if model is None:
        st.error("❌ No trained model found. Please train the model first.")
        return None

    # Load saved feature names
    try:
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)
    except:
        st.error("❌ Feature columns file not found. Re-train the model.")
        return None

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables (same as training)
        input_encoded = pd.get_dummies(input_df)

        # Align with model columns (important!)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0]

        return {
            'prediction': prediction,
            'fraud_probability': probability[1],
            'safe_probability': probability[0]
        }

    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Transaction Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Choose Action:", ["📊 Model Info","🤖 Train Model","🔍 Detect Fraud" ])
    
    if tab == "🔍 Detect Fraud":
        st.header("🔍 Fraud Detection")
        
        # Check if model exists
        if 'model' not in st.session_state and not os.path.exists('trained_model.pkl'):
            st.warning("⚠️ No trained model found. Please train the model first.")
            if st.button("Train Model Now"):
                train_model()
            return
        
        # Input form
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("💰 Transaction Amount", min_value=0.01, value=100.0, step=0.01)
            transaction_type = st.selectbox("💳 Transaction Type", ["purchase", "transfer", "withdrawal", "deposit"])
            channel = st.selectbox("🌐 Transaction Channel", ["online", "mobile_app", "ATM", "branch"])
            is_international = st.selectbox("✈️ Is International Transaction?", ["no", "yes"]) == "yes"
            velocity_1h=st.number_input("⏱️ Velocity (1 Hour)", min_value=0, value=5, step=1)

        with col2:
            avg_7_days = st.number_input("📊 Avg. Transaction Amount (Last 7 Days)", min_value=0.0, value=95.0, step=0.01)
            deviation = st.number_input("📉 Deviation from User Avg Amount", min_value=0.0, value=5.0, step=0.01)
            failed_logins = st.number_input("🚫 Failed Logins Before Transaction",min_value=0, value=0, step=1)
            account_age = st.number_input("📅 Account Age (in days)", min_value=1, value=365, step=1)
            velocity_24h=st.number_input("🚫 Failed Logins Before Transactions  in 24h", min_value=0, value=1, step=1)
            
                
        st.markdown('</div>', unsafe_allow_html=True)
        
       
        if st.button("🔍 Detect Fraud", type="primary"):
            # transaction_amount	transaction_type	channel	is_international	average_transaction_amount_last_7_days	deviation_from_user_average_amount	number_of_failed_logins_before_transaction	account_age	velocity_1h	velocity_24h
            input_data = {
                    'transaction_amount': round(amount, 2),
                    'transaction_type': transaction_type,
                    'channel': channel,
                    'is_international': is_international,
                    'average_transaction_amount_last_7_days': round(avg_7_days, 2),
                    'deviation_from_user_average_amount': round(deviation, 3),
                    'number_of_failed_logins_before_transaction': failed_logins,
                    'account_age': int(account_age),
                    'velocity_1h':int(velocity_1h),
                    'velocity_24h':int(velocity_24h)
                    
                }

            
            # Make prediction
            result = predict_fraud(input_data)
            
            if result:
                
                if result['prediction'] == 1:
                    st.markdown(f"""
                    <div class="prediction-result fraud-alert">
                        🚨 FRAUD DETECTED 🚨<br>
                        Fraud Probability: {result['fraud_probability']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result safe-alert">
                        ✅ TRANSACTION SAFE ✅<br>
                        Safe Probability: {result['safe_probability']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                
                
                st.subheader("📊 Probability Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🔒 Safe Transaction", f"{result['safe_probability']:.2%}")
                
                with col2:
                    st.metric("⚠️ Fraudulent Transaction", f"{result['fraud_probability']:.2%}")
    
    elif tab == "🤖 Train Model":
        st.header("🤖 Model Training")
        st.write("Train the fraud detection model using your dataset.")
        
        if st.button("🚀 Start Training", type="primary"):
            train_model()
        
        # Show current model status
        if 'model' in st.session_state or os.path.exists('trained_model.pkl'):
            st.success("✅ Trained model available")
            if 'metrics' in st.session_state:
                st.subheader("📊 Current Model Performance")
                metrics = st.session_state.metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        else:
            st.info("ℹ️ No trained model found")
    
    elif tab == "📊 Model Info":
        st.header("📊 Model Information")
        
        st.markdown("""
        ### About This System
        This fraud detection system uses machine learning to identify potentially fraudulent transactions.
        
        **Features:**
        - 🤖 Random Forest Classifier
        - 📊 Real-time fraud detection
        - 🎯 High accuracy predictions
        - 📈 Probability scores
        
        **How it works:**
        1. Train the model with historical transaction data
        2. Input new transaction details
        3. Get instant fraud probability assessment
        
        **Model Performance:**
        The system evaluates transactions based on multiple factors including transaction amount, 
        merchant category, time patterns, and user behavior.
        """)

if __name__ == "__main__":
    main()
else:
    st.write("This script is meant to be run as a Streamlit app. Please run it using `streamlit run main.py`.")