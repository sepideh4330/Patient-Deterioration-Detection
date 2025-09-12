"""
Web-based ICU Patient Deterioration Monitor
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import traceback
import os
import sys

# Page configuration
st.set_page_config(
    page_title="ICU Deterioration Monitor",
    page_icon="🏥",
    layout="wide"
)

# --- Main Application Class ---
class ICUMonitorWeb:
    def __init__(self):
        # The app expects a 'models' sub-directory in the same folder as the script
        self.model_path = Path(__file__).parent / "models"
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = None
        self.imputer = None
        self.feature_names = None

    def load_models(self):
        """Load all required models, preprocessing objects, and weights."""
        st.info(f"📁 Attempting to load models from: {self.model_path.absolute()}")
        if not self.model_path.exists():
            st.error(f"❌ Critical Error: Model directory not found at the expected path.")
            st.error("Please create a 'models' folder next to your web_app.py and place all model files inside it.")
            return False

        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
        }
        
        preproc_files = {
            'scaler': 'scaler.pkl',
            'imputer': 'imputer.pkl',
            'weights': 'ensemble_weights.json'
        }

        # Load Models
        for model_key, filename in model_files.items():
            path = self.model_path / filename
            if path.exists():
                try:
                    self.models[model_key] = joblib.load(path)
                    st.success(f"✅ Loaded model: {filename}")
                except Exception as e:
                    st.error(f"❌ Error loading {filename}: {e}")
            else:
                st.warning(f"⚠️ Model file not found: {filename}")
        
        # Load Preprocessing Objects
        scaler_path = self.model_path / preproc_files['scaler']
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            st.success(f"✅ Loaded preprocessor: {preproc_files['scaler']}")
        else:
            st.error(f"❌ Critical: Scaler not found ({preproc_files['scaler']}). Cannot make predictions.")

        imputer_path = self.model_path / preproc_files['imputer']
        if imputer_path.exists():
            self.imputer = joblib.load(imputer_path)
            st.success(f"✅ Loaded preprocessor: {preproc_files['imputer']}")
            if hasattr(self.imputer, 'feature_names_in_'):
                self.feature_names = list(self.imputer.feature_names_in_)
                with st.expander("Model expects the following features"):
                    st.json(self.feature_names)
        else:
            st.error(f"❌ Critical: Imputer not found ({preproc_files['imputer']}). Cannot make predictions.")

        # Load Ensemble Weights
        weights_path = self.model_path / preproc_files['weights']
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)
            st.success(f"✅ Loaded ensemble weights: {self.ensemble_weights}")
        else:
            st.warning(f"⚠️ Weights file not found ({preproc_files['weights']}). Will use a simple average.")

        if not self.models or not self.scaler or not self.imputer:
            st.error("One or more critical components failed to load. The application cannot proceed.")
            return False
            
        return True

    def calculate_scores(self, vitals, labs):
        """Calculate SIRS and qSOFA scores from current data."""
        sirs = (
            (vitals.get('temperature', 37) > 38 or vitals.get('temperature', 37) < 36) +
            (vitals.get('heart_rate', 80) > 90) +
            (vitals.get('resp_rate', 16) > 20) +
            (labs.get('wbc', 10) > 12 or labs.get('wbc', 10) < 4)
        )
        qsofa = (
            (vitals.get('resp_rate', 16) >= 22) +
            (0 < vitals.get('sbp', 120) <= 100)
            # GCS is assumed normal in this simplified UI
        )
        return int(sirs), int(qsofa)


    def prepare_features(self, age, los_hours, vitals, labs):
        """Prepare features matching training pipeline exactly."""
        if self.feature_names is None:
            st.error("Feature names not loaded. Cannot prepare data.")
            return None
        
        # Initialize feature vector with zeros
        feature_dict = {name: 0.0 for name in self.feature_names}
        
        # Set known features
        feature_dict['age'] = age
        if 'gender_male' in feature_dict:
            feature_dict['gender_male'] = 0  # Default
        if 'admission_type_emergency' in feature_dict:
            feature_dict['admission_type_emergency'] = 1  # Default
        
        # For each vital/lab, set the corresponding window features
        # This is simplified - ideally load actual feature mappings
        for window_start, window_end in [(0,6), (6,12), (12,24), (24,36), (36,48)]:
            window_prefix = f'window_{window_start}_{window_end}h'
            
            for vital_name, vital_value in vitals.items():
                for stat in ['mean', 'std', 'min', 'max', 'median', 'count']:
                    feature_name = f'{stat}_{window_prefix}_{vital_name}'
                    if feature_name in feature_dict:
                        if stat == 'count':
                            feature_dict[feature_name] = 10  # Assume 10 measurements
                        elif stat == 'std':
                            feature_dict[feature_name] = vital_value * 0.1  # 10% variation
                        else:
                            feature_dict[feature_name] = vital_value
            
            # Similar for labs
            for lab_name, lab_value in labs.items():
                for stat in ['mean', 'std', 'min', 'max', 'median', 'count']:
                    feature_name = f'{stat}_{window_prefix}_{lab_name}'
                    if feature_name in feature_dict:
                        if stat == 'count':
                            feature_dict[feature_name] = 5  # Fewer lab measurements
                        elif stat == 'std':
                            feature_dict[feature_name] = lab_value * 0.15
                        else:
                            feature_dict[feature_name] = lab_value
        
        # Calculate clinical scores
        sirs, qsofa = self.calculate_scores(vitals, labs)
        if 'sirs_score' in feature_dict:
            feature_dict['sirs_score'] = sirs
        if 'qsofa_score' in feature_dict:
            feature_dict['qsofa_score'] = qsofa
        
        # Create DataFrame with exact column order
        feature_df = pd.DataFrame([feature_dict])[self.feature_names]
        
        # Apply preprocessing
        try:
            feature_array = self.imputer.transform(feature_df)
            feature_array = self.scaler.transform(feature_array)
            return feature_array
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            return None
   
    def predict(self, age, los_hours, vitals, labs):
        """Make predictions using the loaded and prepared models."""
        features = self.prepare_features(age, los_hours, vitals, labs)
        if features is None:
            return None # Stop if feature preparation failed

        predictions = {}
        for name, model in self.models.items():
            try:
                prob = model.predict_proba(features)[0, 1]
                predictions[name] = prob
            except Exception as e:
                st.error(f"Prediction error with {name}: {e}")

        if not predictions:
            return None

        # Calculate ensemble score
        if self.ensemble_weights:
            weighted_sum = sum(predictions.get(name, 0) * weight for name, weight in self.ensemble_weights.items())
            total_weight = sum(weight for name in predictions if name in self.ensemble_weights)
            predictions['ensemble'] = weighted_sum / total_weight if total_weight > 0 else np.mean(list(predictions.values()))
        else:
            predictions['ensemble'] = np.mean(list(predictions.values())) # Fallback

        return predictions

# --- Streamlit UI ---
def main():
    st.title("🏥 AI-Based Early Warning System for ICU Patient Deterioration")
    st.markdown("A real-time risk assessment tool based on the research project by Sepideh Khodadadi.")

    monitor = ICUMonitorWeb()
    
    with st.spinner("Initializing system and loading models..."):
        if not monitor.load_models():
            st.stop()

    # --- Input Section ---
    st.markdown("---")
    st.subheader("Patient Data Input")
    st.info("Enter the patient's most recent measurements. The system will simulate a recent clinical history to generate a risk score.", icon="ℹ️")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 18, 100, 65)
        los_hours = st.slider("ICU Stay (hours)", 1, 720, 48)

    with col2:
        st.markdown("**Vital Signs**")
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 95)
        sbp = st.number_input("Systolic BP (mmHg)", 60, 220, 110)
        dbp = st.number_input("Diastolic BP (mmHg)", 30, 130, 70)
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", 5, 50, 22)
        temperature = st.number_input("Temperature (°C)", 34.0, 42.0, 38.2, format="%.1f")
        spo2 = st.number_input("SpO2 (%)", 70, 100, 93)

    with col3:
        st.markdown("**Key Laboratory Values**")
        wbc = st.number_input("WBC (×10⁹/L)", 0.1, 50.0, 15.0, format="%.1f")
        lactate = st.number_input("Lactate (mmol/L)", 0.1, 20.0, 2.5, format="%.1f")
        creatinine = st.number_input("Creatinine (mg/dL)", 0.1, 15.0, 1.8, format="%.1f")
        glucose = st.number_input("Glucose (mg/dL)", 40, 600, 180)

    vitals = {'heart_rate': heart_rate, 'sbp': sbp, 'dbp': dbp, 'resp_rate': resp_rate, 'temperature': temperature, 'spo2': spo2}
    labs = {'wbc': wbc, 'lactate': lactate, 'creatinine': creatinine, 'glucose': glucose}

    # --- Prediction and Output Section ---
    st.markdown("---")
    if st.button("🔍 Assess Patient Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and predicting risk..."):
            predictions = monitor.predict(age, los_hours, vitals, labs)
        
        if predictions:
            st.subheader("Risk Assessment Results")
            ensemble_risk = predictions['ensemble'] * 100
            
            # Display risk gauge and recommendation
            if ensemble_risk >= 60:
                st.error(f"**🔴 HIGH RISK of Deterioration: {ensemble_risk:.1f}%**")
                st.markdown("> **Recommendation:** Immediate physician assessment and consider rapid response team activation. Review and escalate care plan.")
            elif ensemble_risk >= 30:
                st.warning(f"**🟡 MODERATE RISK of Deterioration: {ensemble_risk:.1f}%**")
                st.markdown("> **Recommendation:** Increase monitoring frequency. Physician assessment recommended within the hour. Review current therapies.")
            else:
                st.success(f"**🟢 LOW RISK of Deterioration: {ensemble_risk:.1f}%**")
                st.markdown("> **Recommendation:** Continue with the standard monitoring and care plan.")

            # Show breakdown of model predictions
            with st.expander("View Individual Model Predictions"):
                data = {'Model': [], 'Risk Prediction (%)': []}
                for name, prob in predictions.items():
                    if name != 'ensemble':
                        data['Model'].append(name.replace("_", " ").title())
                        data['Risk Prediction (%)'].append(prob * 100)
                
                df = pd.DataFrame(data).sort_values(by='Risk Prediction (%)', ascending=False)
                st.dataframe(df.style.format({'Risk Prediction (%)': "{:.1f}"}), use_container_width=True)

if __name__ == "__main__":
    main()