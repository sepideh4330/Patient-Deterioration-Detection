"""
Web-based ICU Patient Deterioration Monitor - Fixed with correct paths
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import time
import traceback
import sys
import os

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="ICU Deterioration Monitor",
    page_icon="🏥",
    layout="wide"
)

class ICUMonitorWeb:
    def __init__(self):
        # Correct path to models - one directory up from Web folder
        self.model_path = Path(r"D:\Dissertation\App\models")
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_models(self):
        """Load models with proper error handling"""
        success = False
        error_messages = []
        
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                st.error(f"❌ Model directory not found: {self.model_path}")
                st.info(f"Current working directory: {os.getcwd()}")
                st.info(f"Looking for models in: {self.model_path.absolute()}")
                return False
            
            st.info(f"📁 Loading models from: {self.model_path}")
            
            # List all files in the models directory
            model_files = list(self.model_path.glob("*.pkl"))
            if model_files:
                st.info(f"Found {len(model_files)} .pkl files")
                for f in model_files:
                    st.text(f"  - {f.name}")
            
            # Load Random Forest
            rf_path = self.model_path / "random_forest.pkl"
            if rf_path.exists():
                try:
                    self.models['random_forest'] = joblib.load(rf_path)
                    st.success(f"✅ Random Forest loaded from {rf_path.name}")
                    success = True
                except Exception as e:
                    st.error(f"❌ Error loading Random Forest: {e}")
            else:
                st.warning(f"⚠️ Random Forest not found at {rf_path}")
            
            # Load XGBoost
            xgb_path = self.model_path / "xgboost.pkl"
            if xgb_path.exists():
                try:
                    self.models['xgboost'] = joblib.load(xgb_path)
                    st.success(f"✅ XGBoost loaded from {xgb_path.name}")
                    success = True
                except Exception as e:
                    st.error(f"❌ Error loading XGBoost: {e}")
            else:
                st.warning(f"⚠️ XGBoost not found at {xgb_path}")
            
            # Load preprocessing objects
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                try:
                    self.scaler = joblib.load(scaler_path)
                    st.success("✅ Scaler loaded")
                except Exception as e:
                    st.warning(f"⚠️ Could not load scaler: {e}")
            else:
                st.warning("⚠️ Scaler not found - using raw features")
            
            imputer_path = self.model_path / "imputer.pkl"
            if imputer_path.exists():
                try:
                    self.imputer = joblib.load(imputer_path)
                    if hasattr(self.imputer, 'feature_names_in_'):
                        self.feature_names = list(self.imputer.feature_names_in_)
                        st.success(f"✅ Imputer loaded with {len(self.feature_names)} features")
                        with st.expander("Expected Features"):
                            st.text(self.feature_names)
                except Exception as e:
                    st.warning(f"⚠️ Could not load imputer: {e}")
            else:
                st.warning("⚠️ Imputer not found - using default features")
            
            if not success:
                st.error("❌ No models could be loaded successfully")
                return False
                
            st.success(f"✅ Successfully loaded {len(self.models)} model(s)")
            return True
            
        except Exception as e:
            st.error(f"❌ Critical error loading models: {str(e)}")
            st.text(traceback.format_exc())
            return False
    
    def calculate_scores(self, vitals, labs):
        """Calculate SIRS and qSOFA scores"""
        sirs = 0
        qsofa = 0
        
        # SIRS Score
        temp = vitals.get('temperature', 37)
        if temp > 38 or temp < 36:
            sirs += 1
        if vitals.get('heart_rate', 80) > 90:
            sirs += 1
        if vitals.get('resp_rate', 16) > 20:
            sirs += 1
        if labs.get('wbc', 10) > 12 or labs.get('wbc', 10) < 4:
            sirs += 1
        
        # qSOFA Score
        if vitals.get('resp_rate', 16) >= 22:
            qsofa += 1
        if 0 < vitals.get('sbp', 120) <= 100:
            qsofa += 1
            
        return sirs, qsofa
    
    def prepare_features(self, age, los_hours, vitals, labs):
        """Prepare feature vector matching your training data structure"""
        # Create comprehensive feature set matching main.py
        features = {
            'AGE': age,
            'LOS_ICU_HOURS': los_hours,
            'MORTALITY_ICU': 0,  # Default values for missing features
            'MORTALITY_HOSP': 0,
            'SEPSIS_FLAG': 0,
            'SEPTIC_SHOCK_FLAG': 0,
        }
        
        # Add vital signs features (matching your main.py aggregation)
        for vital in ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 'temperature']:
            if vital in ['heart_rate']:
                base_val = vitals.get(vital, 80)
            elif vital in ['sbp']:
                base_val = vitals.get(vital, 120)
            elif vital in ['dbp']:
                base_val = vitals.get(vital, 80)
            elif vital in ['map']:
                base_val = vitals.get(vital, 93)
            elif vital in ['resp_rate']:
                base_val = vitals.get(vital, 16)
            elif vital in ['spo2']:
                base_val = vitals.get(vital, 98)
            elif vital in ['temperature']:
                base_val = vitals.get(vital, 37)
            else:
                base_val = 0
            
            features[f'mean_{vital}'] = base_val
            features[f'std_{vital}'] = base_val * 0.1  # Approximate std as 10% of mean
            features[f'min_{vital}'] = base_val * 0.9
            features[f'max_{vital}'] = base_val * 1.1
        
        # Add lab features
        for lab in ['creatinine', 'bun', 'wbc', 'hemoglobin', 'platelet', 'sodium', 
                   'potassium', 'chloride', 'bicarbonate', 'lactate', 'glucose', 'bilirubin']:
            if lab in labs:
                base_val = labs[lab]
            else:
                # Default values
                defaults = {
                    'creatinine': 1.0, 'bun': 20, 'wbc': 10, 'hemoglobin': 14,
                    'platelet': 250, 'sodium': 140, 'potassium': 4.0,
                    'chloride': 100, 'bicarbonate': 24, 'lactate': 1.0,
                    'glucose': 100, 'bilirubin': 1.0
                }
                base_val = defaults.get(lab, 0)
            
            features[f'mean_{lab}'] = base_val
            features[f'min_{lab}'] = base_val * 0.9
            features[f'max_{lab}'] = base_val * 1.1
            features[f'change_{lab}'] = 0  # No change for single measurement
        
        # Add severity scores
        sirs, qsofa = self.calculate_scores(vitals, labs)
        features['SIRS_SCORE'] = sirs
        features['QSOFA_SCORE'] = qsofa
        
        # Add encoded categorical variables (with default values)
        features['ADMISSION_TYPE_encoded'] = 0  # Emergency
        features['INSURANCE_encoded'] = 0  # Default insurance
        features['ETHNICITY_encoded'] = 0  # Default ethnicity
        features['GENDER_encoded'] = 0  # Default gender
        
        # Create DataFrame
        feature_df = pd.DataFrame([features])
        
        # If we know the expected features, align them
        if self.feature_names is not None:
            # Add any missing features with default value 0
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            # Select only the features the model expects
            feature_df = feature_df[self.feature_names]
        
        # Apply preprocessing
        try:
            if self.imputer is not None:
                feature_array = self.imputer.transform(feature_df)
            else:
                feature_array = feature_df.fillna(0).values
                
            if self.scaler is not None:
                feature_array = self.scaler.transform(feature_array)
                
            return feature_array
            
        except Exception as e:
            st.error(f"Feature preparation error: {e}")
            return feature_df.fillna(0).values
    
    def predict(self, age, los_hours, vitals, labs):
        """Make predictions using loaded models"""
        if not self.models:
            st.error("No models available for prediction")
            return None
        
        try:
            features = self.prepare_features(age, los_hours, vitals, labs)
            st.info(f"Prepared {features.shape[1] if len(features.shape) > 1 else len(features)} features")
            
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    prob = model.predict_proba(features)[0, 1]
                    predictions[model_name] = prob
                except Exception as e:
                    st.error(f"Prediction error with {model_name}: {e}")
            
            if predictions:
                predictions['ensemble'] = np.mean(list(predictions.values()))
            
            return predictions
            
        except Exception as e:
            st.error(f"Overall prediction error: {e}")
            st.text(traceback.format_exc())
            return None

def main():
    st.title("🏥 ICU Patient Deterioration Monitor")
    st.markdown("### Real-time Risk Assessment System")
    
    # Initialize monitor
    monitor = ICUMonitorWeb()
    
    # Load models
    with st.expander("System Status", expanded=True):
        if not monitor.load_models():
            st.error("Please check that your models are in: D:\\Dissertation\\App\\models")
            st.stop()
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Input section
    st.markdown("---")
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 18, 100, 65)
        los_hours = st.slider("ICU Stay (hours)", 0, 720, 24)
        
    with col2:
        st.markdown("**Vital Signs**")
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 80)
        sbp = st.number_input("Systolic BP (mmHg)", 70, 200, 120)
        dbp = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
        resp_rate = st.number_input("Respiratory Rate", 8, 40, 16)
        temperature = st.number_input("Temperature (°C)", 35.0, 40.0, 37.0)
        spo2 = st.number_input("SpO2 (%)", 70, 100, 98)
        
    with col3:
        st.markdown("**Laboratory Values**")
        wbc = st.number_input("WBC (×10⁹/L)", 0.1, 50.0, 10.0)
        lactate = st.number_input("Lactate (mmol/L)", 0.1, 20.0, 1.0)
        creatinine = st.number_input("Creatinine (mg/dL)", 0.1, 10.0, 1.0)
        glucose = st.number_input("Glucose (mg/dL)", 50, 500, 100)
    
    # Prepare data
    vitals = {
        'heart_rate': heart_rate,
        'sbp': sbp,
        'dbp': dbp,
        'map': dbp + (sbp - dbp) / 3,
        'resp_rate': resp_rate,
        'temperature': temperature,
        'spo2': spo2
    }
    
    labs = {
        'wbc': wbc,
        'lactate': lactate,
        'creatinine': creatinine,
        'glucose': glucose
    }
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            predictions = monitor.predict(age, los_hours, vitals, labs)
            
            if predictions:
                risk = predictions.get('ensemble', 0) * 100
                
                # Display results
                st.markdown("---")
                st.subheader("Risk Assessment Results")
                
                # Risk level
                if risk >= 70:
                    st.error(f"🔴 **HIGH RISK**: {risk:.1f}%")
                    st.warning("Immediate intervention recommended")
                elif risk >= 40:
                    st.warning(f"🟡 **MODERATE RISK**: {risk:.1f}%")
                    st.info("Close monitoring advised")
                else:
                    st.success(f"🟢 **LOW RISK**: {risk:.1f}%")
                    st.info("Continue routine care")
                
                # Model breakdown
                if len(predictions) > 1:
                    st.markdown("**Model Predictions:**")
                    for model, prob in predictions.items():
                        if model != 'ensemble':
                            st.text(f"  • {model}: {prob*100:.1f}%")

if __name__ == "__main__":
    main()