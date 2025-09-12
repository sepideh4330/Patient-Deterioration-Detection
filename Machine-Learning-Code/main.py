# =============================================================================
#
# ICU DETERIORATION PREDICTION
#
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import os
import sys
from tqdm import tqdm
import pickle
import gc
import multiprocessing as mp
from pathlib import Path
import json
import h5py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, confusion_matrix, recall_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             accuracy_score, f1_score, classification_report,
                             precision_score, matthews_corrcoef, cohen_kappa_score,
                             brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.preprocessing import RobustScaler
# ### MODIFICATION ### - Using SimpleImputer for speed. KNNImputer is commented out.
from sklearn.impute import SimpleImputer, KNNImputer 
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, Attention, Input, Concatenate,
                                     Conv1D, GlobalMaxPooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Statistical tests
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import ttest_ind
from sklearn.base import BaseEstimator, ClassifierMixin

# Interpretability
import shap

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """configuration combining best parameters from both implementations."""

    # Data paths
    DATA_PATH: Path = Path('./mimic-iii-clinical-database-1.4')
    OUTPUT_PATH: Path = Path('./output')
    CACHE_PATH: Path = Path('./cache')

    # Processing parameters
    CHUNK_SIZE: int = 5_000_000
    N_JOBS: int = max(1, mp.cpu_count() - 2)
    RANDOM_STATE: int = 42

    # Temporal windows - Multi-window approach
    OBSERVATION_WINDOWS: list = field(default_factory=lambda: [
        (0, 6), (6, 12), (12, 24), (24, 36), (36, 48)
    ])
    GAP_WINDOW: Tuple[int, int] = (48, 50)
    PREDICTION_WINDOW: Tuple[int, int] = (50, 74)

    # Feature engineering
    MIN_MEASUREMENTS_PER_PATIENT: int = 10
    MAX_MISSING_RATE: float = 0.8

    # Model parameters
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15

    # Comprehensive MIMIC-III Item IDs
    VITAL_SIGNS_ITEMS: dict = field(default_factory=lambda: {
        'heart_rate': [211, 220045],
        'sbp': [51, 442, 455, 6701, 220050, 220179],
        'dbp': [8368, 8440, 8441, 8555, 220051, 220180],
        'mbp': [456, 52, 6702, 443, 220052, 220181],
        'resp_rate': [615, 618, 220210, 224690],
        'temperature': [223761, 678, 223762, 676],
        'spo2': [646, 220277],
        'glucose': [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
    })

    LAB_ITEMS: dict = field(default_factory=lambda: {
        'lactate': [50813],
        'creatinine': [50912],
        'bun': [51006],
        'sodium': [50824, 50983],
        'potassium': [50822, 50971],
        'chloride': [50806, 50902],
        'bicarbonate': [50882],
        'hemoglobin': [51222, 50811],
        'hematocrit': [51221, 50810],
        'wbc': [51300, 51301],
        'platelet': [51265],
        'inr': [51237],
        'ptt': [51274],
        'bilirubin': [50885],
        'albumin': [50862],
    })

    NEUROLOGICAL_ITEMS: dict = field(default_factory=lambda: {
        'gcs_total': [198],
        'gcs_motor': [454],
        'gcs_verbal': [723],
        'gcs_eyes': [184],
    })

    URINE_ITEMS: dict = field(default_factory=lambda: {
        'urine_output': [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 40405, 40428, 40086, 40096, 40651]
    })

    def get_all_item_ids(self) -> List[int]:
        """Get all item IDs as a flat list."""
        all_items = []
        for item_dict in [self.VITAL_SIGNS_ITEMS, self.LAB_ITEMS,
                          self.NEUROLOGICAL_ITEMS, self.URINE_ITEMS]:
            for items in item_dict.values():
                all_items.extend(items)
        return list(set(all_items))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_logging(output_path: Path) -> logging.Logger:
    """Setup comprehensive logging with both file and console output."""
    log_path = output_path / 'logs'
    log_path.mkdir(exist_ok=True, parents=True)

    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(
        log_path / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

def memory_usage() -> float:
    """Get current memory usage in GB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def clean_memory():
    """Force garbage collection."""
    gc.collect()

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class DataLoader:
    """data loading for large MIMIC-III files with caching."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_core_tables(self) -> Dict[str, pd.DataFrame]:
        """Load core MIMIC-III tables with dtypes."""
        self.logger.info("Loading core MIMIC-III tables...")
        tables = {}
        dtype_specs = {
            'ADMISSIONS.csv.gz': {
                'SUBJECT_ID': 'int32', 'HADM_ID': 'int32', 'ADMISSION_TYPE': 'category',
                'INSURANCE': 'category', 'ETHNICITY': 'category', 'DIAGNOSIS': 'str'
            },
            'PATIENTS.csv.gz': {'SUBJECT_ID': 'int32', 'GENDER': 'category'},
            'ICUSTAYS.csv.gz': {
                'SUBJECT_ID': 'int32', 'HADM_ID': 'int32', 'ICUSTAY_ID': 'int32',
                'FIRST_CAREUNIT': 'category', 'LAST_CAREUNIT': 'category'
            }
        }
        date_cols_map = {
            'ADMISSIONS.csv.gz': ['ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'EDREGTIME', 'EDOUTTIME'],
            'PATIENTS.csv.gz': ['DOB', 'DOD', 'DOD_HOSP', 'DOD_SSN'],
            'ICUSTAYS.csv.gz': ['INTIME', 'OUTTIME']
        }

        for file_name, dtypes in dtype_specs.items():
            file_path = self.config.DATA_PATH / file_name
            self.logger.info(f"  Loading {file_name}...")
            df = pd.read_csv(file_path, dtype=dtypes)
            for col in date_cols_map.get(file_name, []):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            table_name = file_name.replace('.csv.gz', '').lower()
            tables[table_name] = df
            self.logger.info(f"    Loaded {len(df):,} rows, Memory: {memory_usage():.2f} GB")
        return tables

    def create_cohort(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create study cohort with inclusion/exclusion criteria."""
        self.logger.info("Creating study cohort...")
        cohort = tables['admissions'].merge(tables['patients'], on='SUBJECT_ID')
        cohort = cohort.merge(tables['icustays'], on=['SUBJECT_ID', 'HADM_ID'])
        cohort['AGE'] = cohort['INTIME'].dt.year - cohort['DOB'].dt.year
        cohort.loc[cohort['AGE'] > 89, 'AGE'] = 90
        initial_count = len(cohort)
        self.logger.info(f"  Initial admissions: {initial_count:,}")
        cohort = cohort[cohort['AGE'] >= 18]
        self.logger.info(f"  After age >= 18: {len(cohort):,}")
        cohort = cohort.sort_values(['SUBJECT_ID', 'INTIME']).drop_duplicates(subset='SUBJECT_ID', keep='first')
        self.logger.info(f"  After first admission only: {len(cohort):,}")
        cohort['LOS_HOURS'] = (cohort['OUTTIME'] - cohort['INTIME']).dt.total_seconds() / 3600
        cohort = cohort[cohort['LOS_HOURS'] >= 74]
        self.logger.info(f"  After LOS >= 74h: {len(cohort):,}")
        cohort = cohort[~cohort['DIAGNOSIS'].str.contains('COMFORT CARE', case=False, na=False)]
        self.logger.info(f"  Final cohort: {len(cohort):,} patients")
        cohort['MORTALITY_ICU'] = cohort['DEATHTIME'].notna() & (cohort['DEATHTIME'] <= cohort['OUTTIME'])
        return cohort

    def process_events_parallel(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Process chartevents and labevents in parallel for efficiency."""
        self.logger.info("Processing events data (this will take time)...")
        
        # ### MODIFICATION ### Using faster Parquet format for caching
        cache_path = self.config.CACHE_PATH / 'events_processed.parquet'
        cache_path.parent.mkdir(exist_ok=True, parents=True)

        if cache_path.exists():
            self.logger.info("  Loading from cache...")
            return pd.read_parquet(cache_path)

        icustay_ids = set(cohort['ICUSTAY_ID'].values)
        item_ids = set(self.config.get_all_item_ids())
        all_events = []

        for file_name in ['CHARTEVENTS.csv.gz', 'LABEVENTS.csv.gz']:
            file_path = self.config.DATA_PATH / file_name
            if not file_path.exists():
                self.logger.warning(f"  {file_name} not found, skipping...")
                continue

            self.logger.info(f"  Processing {file_name}...")
            chunks_processed = []
            
            # ### MODIFICATION ### Prepare for optimized merging for LABEVENTS
            lab_chunks = []
            
            usecols = ['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'] if 'CHART' in file_name else ['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']
            
            with tqdm(desc=f"Processing {file_name}") as pbar:
                for chunk_num, chunk in enumerate(pd.read_csv(
                    file_path, chunksize=self.config.CHUNK_SIZE, usecols=usecols,
                    dtype={'ITEMID': 'int32', 'VALUENUM': 'float32'}
                )):
                    chunk = chunk[(chunk['ITEMID'].isin(item_ids)) & (chunk['VALUENUM'].notna())]
                    
                    if 'HADM_ID' in chunk.columns:
                        # ### MODIFICATION ### Just append chunk, merge will happen ONCE later
                        lab_chunks.append(chunk)
                    elif 'ICUSTAY_ID' in chunk.columns:
                        chunk = chunk[chunk['ICUSTAY_ID'].isin(icustay_ids)]
                        if not chunk.empty:
                            chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'])
                            chunks_processed.append(chunk[['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']])
                    
                    pbar.update(1)
                    if chunk_num % 10 == 0: clean_memory()
            
            # ### MODIFICATION ### Process all collected LABEVENTS chunks at once
            if lab_chunks:
                self.logger.info("  Post-processing LABEVENTS...")
                lab_df = pd.concat(lab_chunks, ignore_index=True)
                hadm_to_icustay = cohort[['HADM_ID', 'ICUSTAY_ID']].drop_duplicates()
                lab_df = lab_df.merge(hadm_to_icustay, on='HADM_ID', how='inner')
                lab_df = lab_df[lab_df['ICUSTAY_ID'].isin(icustay_ids)]
                if not lab_df.empty:
                    lab_df['CHARTTIME'] = pd.to_datetime(lab_df['CHARTTIME'])
                    all_events.append(lab_df[['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM']])

            if chunks_processed: # This handles CHARTEVENTS
                all_events.extend(chunks_processed)

        self.logger.info("  Combining all events...")
        events = pd.concat(all_events, ignore_index=True)

        item_map = {}
        for item_dict in [self.config.VITAL_SIGNS_ITEMS, self.config.LAB_ITEMS, self.config.NEUROLOGICAL_ITEMS, self.config.URINE_ITEMS]:
            for name, ids in item_dict.items():
                for item_id in ids:
                    item_map[item_id] = name
        events['item_name'] = events['ITEMID'].map(item_map)

        self.logger.info("  Saving to cache...")
        # ### MODIFICATION ### Using Parquet
        events.to_parquet(cache_path)
        
        self.logger.info(f"  Final events: {len(events):,} rows")
        return events

# =============================================================================
# FEATURE ENGINEERING (Original class kept for helper methods)
# =============================================================================

class FeatureEngineer:
    """Feature engineering class, now primarily for helper functions used by the vectorized process."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # NOTE: The main feature creation logic is now in ICUPipeline._create_features_vectorized
    # The methods below can be adapted or used to calculate more complex features after initial aggregation.

    def _get_aggregated_value(self, features: Dict[str, float], item_name: str, stat: str = 'mean') -> float:
        """Helper function to get aggregated value across all observation windows."""
        values = []
        for start, end in self.config.OBSERVATION_WINDOWS:
            key = f'window_{start}_{end}h_{item_name}_{stat}'
            if key in features and pd.notna(features[key]):
                values.append(features[key])
        return np.nanmean(values) if values else np.nan

    def _calculate_standard_clinical_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard clinical severity scores on an aggregated feature DataFrame."""
        self.logger.info("Calculating clinical scores (SIRS, qSOFA)...")
        
        # Helper to get the mean value of a feature across all windows
        def get_mean_feature(item_name, stat='mean'):
            cols = [c for c in features.columns if f"{item_name}_{stat}" in c]
            if not cols: return pd.Series(np.nan, index=features.index)
            return features[cols].mean(axis=1)

        temp = get_mean_feature('temperature')
        hr = get_mean_feature('heart_rate')
        rr = get_mean_feature('resp_rate')
        wbc = get_mean_feature('wbc')
        sbp = get_mean_feature('sbp')
        gcs = get_mean_feature('gcs_total')

        sirs_score = ((temp > 38) | (temp < 36)).astype(int) + \
                     (hr > 90).astype(int) + \
                     (rr > 20).astype(int) + \
                     ((wbc > 12) | (wbc < 4)).astype(int)
        
        qsofa_score = (rr >= 22).astype(int) + \
                      (sbp <= 100).astype(int) + \
                      (gcs < 15).astype(int)
                      
        features['sirs_score'] = sirs_score
        features['qsofa_score'] = qsofa_score
        return features

# =============================================================================
# MODEL TRAINING AND OPTIMIZATION
# =============================================================================

class ModelTrainer:
    """model training with hyperparameter optimization and calibration."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.uncalibrated_models = {}

    def train_balanced_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray) -> Any:
        self.logger.info("Training Balanced Random Forest...")
        param_distributions = {
            'n_estimators': [100, 200, 300], 'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        rf = BalancedRandomForestClassifier(random_state=self.config.RANDOM_STATE, n_jobs=self.config.N_JOBS)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.RANDOM_STATE)
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=cv, scoring='roc_auc', 
            n_jobs=self.config.N_JOBS, verbose=1, random_state=self.config.RANDOM_STATE
        )
        random_search.fit(X_train, y_train)
        self.logger.info(f"  Best parameters: {random_search.best_params_}")
        self.logger.info(f"  Best CV score: {random_search.best_score_:.4f}")
        val_score = roc_auc_score(y_val, random_search.predict_proba(X_val)[:, 1])
        self.logger.info(f"  Validation AUC-ROC: {val_score:.4f}")
        self.uncalibrated_models['random_forest'] = random_search.best_estimator_
        return random_search.best_estimator_

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Any:
        self.logger.info("Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params = {
            'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05,
            'subsample': 0.85, 'colsample_bytree': 0.85, 'gamma': 0.1,
            'reg_alpha': 0.01, 'reg_lambda': 1.5, 'scale_pos_weight': scale_pos_weight,
            'random_state': self.config.RANDOM_STATE, 'n_jobs': self.config.N_JOBS,
            'tree_method': 'hist', 'eval_metric': 'auc', 'early_stopping_rounds': 20,
        }
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_score = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
        self.logger.info(f"  Best iteration: {xgb_model.best_iteration}")
        self.logger.info(f"  Validation AUC-ROC: {val_score:.4f}")
        self.uncalibrated_models['xgboost'] = xgb_model
        return xgb_model

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        self.logger.info("Training LightGBM...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = LGBMClassifier(
            objective='binary',
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=self.config.RANDOM_STATE,
            n_jobs=self.config.N_JOBS,
            scale_pos_weight=scale_pos_weight,
            verbose=-1
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        val_pred = model.predict_proba(X_val)[:, 1]
        val_score = roc_auc_score(y_val, val_pred)
        self.logger.info(f"  Best iteration: {model.best_iteration_}")
        self.logger.info(f"  Validation AUC-ROC: {val_score:.4f}")
        self.uncalibrated_models['lightgbm'] = model
        return model

    def train_lstm_attention(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Any:
        self.logger.info("Training LSTM with Attention...")
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

        inputs = Input(shape=(1, X_train.shape[1]))
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(inputs)
        lstm1 = BatchNormalization()(lstm1)
        attention = Attention()([lstm1, lstm1])
        concat = Concatenate()([lstm1, attention])
        lstm2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(concat)
        lstm2 = BatchNormalization()(lstm2)
        dense1 = Dense(32, activation='relu')(lstm2)
        dense1 = Dropout(0.3)(dense1)
        dense2 = Dense(16, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        outputs = Dense(1, activation='sigmoid')(dense2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

        callbacks = [
            EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(str(self.config.OUTPUT_PATH / 'best_lstm.h5'), monitor='val_auc', mode='max', save_best_only=True)
        ]
        class_weight = {0: 1, 1: (y_train == 0).sum() / (y_train == 1).sum()}
        history = model.fit(
            X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
            epochs=100, batch_size=32, callbacks=callbacks, class_weight=class_weight, verbose=0
        )
        val_pred = model.predict(X_val_lstm, verbose=0).flatten()
        val_score = roc_auc_score(y_val, val_pred)
        self.logger.info(f"  Best epoch: {np.argmax(history.history['val_auc']) + 1}")
        self.logger.info(f"  Validation AUC-ROC: {val_score:.4f}")
        self.uncalibrated_models['lstm'] = model
        return model, history

    def calibrate_models(self, X_val: np.ndarray, y_val: np.ndarray):
        self.logger.info("Calibrating models...")
        for name in ['random_forest', 'xgboost']:
            if name in self.uncalibrated_models:
                self.models[name] = CalibratedClassifierCV(self.uncalibrated_models[name], method='isotonic', cv='prefit')
                self.models[name].fit(X_val, y_val)
                self.logger.info(f"  Calibrated {name}")

        if 'lightgbm' in self.uncalibrated_models:
            self.models['lightgbm'] = CalibratedClassifierCV(
                self.uncalibrated_models['lightgbm'],
                method='isotonic',
                cv='prefit'
            )
            self.models['lightgbm'].fit(X_val, y_val)
            self.logger.info("  Calibrated lightgbm")
  
        if 'lstm' in self.uncalibrated_models:
            self.models['lstm'] = self.uncalibrated_models['lstm']

    def create_ensemble(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        self.logger.info("Creating ensemble...")
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm':
                X_val_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                predictions[name] = model.predict(X_val_lstm, verbose=0).flatten()
            else:
                predictions[name] = model.predict_proba(X_val)[:, 1]

        def ensemble_loss(weights):
            weights = weights / np.sum(weights)
            weighted_pred = np.zeros_like(y_val, dtype=float)
            for i, name in enumerate(predictions.keys()):
                weighted_pred += weights[i] * predictions[name]
            return -roc_auc_score(y_val, weighted_pred)

        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1)] * n_models
        result = minimize(ensemble_loss, initial_weights, bounds=bounds, method='SLSQP')
        optimal_weights = result.x / np.sum(result.x)
        ensemble_weights = dict(zip(predictions.keys(), optimal_weights))

        for name, weight in ensemble_weights.items():
            self.logger.info(f"  {name}: {weight:.3f}")

        ensemble_pred = np.zeros_like(y_val, dtype=float)
        for name, weight in ensemble_weights.items():
            ensemble_pred += weight * predictions[name]
        ensemble_score = roc_auc_score(y_val, ensemble_pred)
        self.logger.info(f"  Ensemble validation AUC-ROC: {ensemble_score:.4f}")
        return ensemble_weights

# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

class ComprehensiveEvaluator:
    # This class remains unchanged as it was already well-structured.
    """Comprehensive model evaluation with statistical analysis and visualization."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}

    def evaluate_all_models(self, models: Dict, ensemble_weights: Dict,
                            X_test: np.ndarray, y_test: np.ndarray,
                            feature_names: List[str]) -> Dict:
        self.logger.info("Evaluating all models on test set...")
        predictions = self._get_all_predictions(models, ensemble_weights, X_test, y_test)
        for name, y_pred_proba in predictions.items():
            self.results[name] = self._calculate_comprehensive_metrics(y_test, y_pred_proba)
            self.logger.info(f"\n{name.upper()} Performance:")
            self.logger.info(f"  AUC-ROC: {self.results[name]['auc_roc']:.4f}")
            self.logger.info(f"  AUC-PR: {self.results[name]['auc_pr']:.4f}")
            self.logger.info(f"  F1 Score: {self.results[name]['f1']:.4f}")
            self.logger.info(f"  Sensitivity: {self.results[name]['sensitivity']:.4f}")
            self.logger.info(f"  Specificity: {self.results[name]['specificity']:.4f}")
            self.logger.info(f"  Brier Score: {self.results[name]['brier']:.4f}")
        self._statistical_comparison(predictions, y_test)
        self._analyze_feature_importance(models, X_test, feature_names)
        self._generate_all_plots(predictions, y_test)
        self._generate_final_report()
        return self.results

    def _get_all_predictions(self, models: Dict, ensemble_weights: Dict,
                             X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        predictions = {}
        for name, model in models.items():
            if name == 'lstm':
                X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                predictions[name] = model.predict(X_test_lstm, verbose=0).flatten()
            else:
                predictions[name] = model.predict_proba(X_test)[:, 1]
        ensemble_pred = np.zeros(len(y_test))
        for name, weight in ensemble_weights.items():
            ensemble_pred += weight * predictions[name]
        predictions['ensemble'] = ensemble_pred
        return predictions

    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba), 'auc_pr': average_precision_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred), 'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0, 'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0, 'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'brier': brier_score_loss(y_true, y_pred_proba), 'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }
        return metrics

    def _statistical_comparison(self, predictions: Dict, y_test: np.ndarray):
        self.logger.info("\n\nStatistical Comparison (Bootstrap test for AUC difference):")
        ensemble_pred = predictions['ensemble']
        for name, pred in predictions.items():
            if name != 'ensemble':
                n_bootstrap = 1000
                auc_diffs = []
                for _ in range(n_bootstrap):
                    idx = np.random.choice(len(y_test), len(y_test), replace=True)
                    try:
                        auc_ensemble = roc_auc_score(y_test[idx], ensemble_pred[idx])
                        auc_model = roc_auc_score(y_test[idx], pred[idx])
                        auc_diffs.append(auc_ensemble - auc_model)
                    except ValueError: continue
                if auc_diffs:
                    mean_diff = np.mean(auc_diffs)
                    ci_lower, ci_upper = np.percentile(auc_diffs, 2.5), np.percentile(auc_diffs, 97.5)
                    p_value = 2 * min(np.mean(np.array(auc_diffs) <= 0), np.mean(np.array(auc_diffs) >= 0))
                    self.logger.info(f"  Ensemble vs {name}:")
                    self.logger.info(f"    AUC difference: {mean_diff:.4f} (95% CI: {ci_lower:.4f} to {ci_upper:.4f})")
                    self.logger.info(f"    p-value: {p_value:.4f}")

    def _analyze_feature_importance(self, models: Dict, X_test: np.ndarray, feature_names: List[str]):
        self.logger.info("\n\nAnalyzing feature importance...")
        # Using SHAP for more robust feature importance
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        for name, model in models.items():
            if name == 'lstm': continue # SHAP for LSTM is more complex
            
            try:
                if name == 'lightgbm':
                     # Need to get the base model from the wrapper
                      base_model = model.calibrated_classifiers_[0].base_estimator
                      explainer = shap.TreeExplainer(
                            base_model.booster_ if hasattr(base_model, "booster_") else base_model
                        )
                else:
                    base_model = model.calibrated_classifiers_[0].base_estimator
                    explainer = shap.TreeExplainer(base_model)
                
                shap_values = explainer.shap_values(X_test_df)
                
                # For binary classification, shap_values can be a list of two arrays
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] # Use values for the positive class
                
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False)
                
                shap_df.to_csv(self.config.OUTPUT_PATH / f'feature_importance_{name}_shap.csv', index=False)
                self.logger.info(f"  Top 5 SHAP features for {name}:")
                for _, row in shap_df.head(5).iterrows():
                    self.logger.info(f"    {row['feature']}: {row['importance']:.4f}")

            except Exception as e:
                self.logger.warning(f"Could not compute SHAP values for {name}: {e}")

    def _generate_all_plots(self, predictions: Dict, y_test: np.ndarray):
        output_path = self.config.OUTPUT_PATH / 'plots'
        output_path.mkdir(exist_ok=True, parents=True)
        self._plot_roc_curves(predictions, y_test, output_path)
        self._plot_pr_curves(predictions, y_test, output_path)
        self._plot_calibration(predictions, y_test, output_path)
        self._plot_performance_comparison(output_path)
        self.logger.info(f"\n\nAll plots saved to {output_path}")

    def _plot_roc_curves(self, predictions: Dict, y_test: np.ndarray, output_path: Path):
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for (name, y_pred), color in zip(predictions.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{name.upper()} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12); plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_pr_curves(self, predictions: Dict, y_test: np.ndarray, output_path: Path):
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        baseline = y_test.mean()
        for (name, y_pred), color in zip(predictions.items(), colors):
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            auc_pr = average_precision_score(y_test, y_pred)
            plt.plot(recall, precision, color=color, lw=2, label=f'{name.upper()} (AP = {auc_pr:.3f})')
        plt.axhline(y=baseline, color='k', linestyle='--', lw=1, label=f'Baseline (AP = {baseline:.3f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(output_path / 'pr_curves.png', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_calibration(self, predictions: Dict, y_test: np.ndarray, output_path: Path):
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        for (name, y_pred), color in zip(predictions.items(), colors):
            fraction_pos, mean_pred = calibration_curve(y_test, y_pred, n_bins=10)
            plt.plot(mean_pred, fraction_pos, marker='o', linewidth=2, label=f'{name.upper()}', color=color)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability', fontsize=12); plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Plots', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(output_path / 'calibration_plots.png', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_performance_comparison(self, output_path: Path):
        metrics_to_plot = ['auc_roc', 'auc_pr', 'f1', 'sensitivity', 'specificity', 'mcc']
        data = [{'Model': model.upper(), 'Metric': metric.upper().replace('_', ' '), 'Score': metrics[metric]}
                for model, metrics in self.results.items() for metric in metrics_to_plot if metric in metrics]
        df = pd.DataFrame(data)
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df, x='Model', y='Score', hue='Metric', palette='viridis')
        plt.ylabel('Score', fontsize=12); plt.xlabel('Model', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right'); plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3); plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight'); plt.close()
        
    def _generate_final_report(self):
        report_path = self.config.OUTPUT_PATH / 'FINAL_REPORT.txt'
        self.logger.info(f"Generating final report at: {report_path}")
        with open(report_path, 'w') as f:
            f.write("="*80 + "\nICU DETERIORATION PREDICTION PIPELINE REPORT\n" + "="*80 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("MODEL PERFORMANCE SUMMARY ON TEST SET\n" + "-"*40 + "\n\n")
            sorted_models = sorted(self.results.items(), key=lambda x: x[1]['auc_roc'], reverse=True)
            for model_name, metrics in sorted_models:
                f.write(f"--- {model_name.upper()} ---\n")
                f.write(f"  AUC-ROC:         {metrics.get('auc_roc', 0):.4f}\n")
                f.write(f"  AUC-PR:          {metrics.get('auc_pr', 0):.4f}\n")
                f.write(f"  Brier Score:     {metrics.get('brier', 0):.4f}\n")
                f.write(f"  F1 Score:        {metrics.get('f1', 0):.4f}\n")
                f.write(f"  Sensitivity:     {metrics.get('sensitivity', 0):.4f}\n")
                f.write(f"  Specificity:     {metrics.get('specificity', 0):.4f}\n")
                f.write(f"  PPV (Precision): {metrics.get('ppv', 0):.4f}\n")
                f.write(f"  NPV:             {metrics.get('npv', 0):.4f}\n")
                f.write(f"  MCC:             {metrics.get('mcc', 0):.4f}\n\n")
            f.write("\n" + "="*80 + "\nEND OF REPORT\n" + "="*80 + "\n")
        self.logger.info("Final report generated successfully.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class ICUPipeline:
    """Main pipeline orchestrating all components."""

    def __init__(self):
        self.config = Config()
        self.config.OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
        self.config.CACHE_PATH.mkdir(exist_ok=True, parents=True)

        self.logger = setup_logging(self.config.OUTPUT_PATH)
        self.logger.info("="*80)
        self.logger.info("ICU DETERIORATION PREDICTION - ENHANCED HYBRID PIPELINE")
        self.logger.info("="*80)

        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.evaluator = ComprehensiveEvaluator(self.config)
        
        self.cohort = None
        self.features_df = None
        self.models = {}

    def run(self):
        """Execute complete pipeline."""
        try:
            self.logger.info("\n" + "="*60 + "\nSTEP 1: DATA LOADING AND COHORT CREATION\n" + "="*60)
            tables = self.data_loader.load_core_tables()
            self.cohort = self.data_loader.create_cohort(tables)
            events_df = self.data_loader.process_events_parallel(self.cohort)
            
            self.logger.info("\n" + "="*60 + "\nSTEP 2: FEATURE ENGINEERING\n" + "="*60)
            # ### MODIFICATION ### Calling the new vectorized function
            self._create_features_vectorized(events_df)
            
            self.logger.info("\n" + "="*60 + "\nSTEP 3: DATA PREPARATION\n" + "="*60)
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = self._prepare_data()
            
            self.logger.info("\n" + "="*60 + "\nSTEP 4: MODEL TRAINING AND CALIBRATION\n" + "="*60)
            self._train_models(X_train, y_train, X_val, y_val)

            self.logger.info("\n" + "="*60 + "\nSTEP 5: ENSEMBLE OPTIMIZATION\n" + "="*60)
            ensemble_weights = self.model_trainer.create_ensemble(X_val, y_val)
            
            self.logger.info("\n" + "="*60 + "\nSTEP 6: COMPREHENSIVE EVALUATION\n" + "="*60)
            results = self.evaluator.evaluate_all_models(
                self.models, ensemble_weights, X_test, y_test, feature_names
            )
            
            self._save_models_and_results(ensemble_weights, results)
            
            self.logger.info("\n" + "="*80 + "\nPIPELINE COMPLETED SUCCESSFULLY!\n" + "="*80)
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    ### NEW / REFACTORED ###
    def _create_features_vectorized(self, events_df: pd.DataFrame):
        """A vectorized approach to feature engineering using groupby operations."""
        self.logger.info("Starting vectorized feature engineering...")

        # ========== 1. Prepare Events DataFrame ==========
        self.logger.info("  Merging cohort data and calculating time from admission...")
        events_df = events_df.merge(
            self.cohort[['ICUSTAY_ID', 'INTIME', 'DEATHTIME']], on='ICUSTAY_ID', how='left'
        )
        events_df['hours_from_admission'] = (events_df['CHARTTIME'] - events_df['INTIME']).dt.total_seconds() / 3600

        self.logger.info("  Filtering patients by minimum measurement count...")
        obs_events = events_df[events_df['hours_from_admission'] < 48].copy()
        patient_counts = obs_events.groupby('ICUSTAY_ID')['ITEMID'].count()
        valid_icustay_ids = patient_counts[patient_counts >= self.config.MIN_MEASUREMENTS_PER_PATIENT].index
        
        events_df = events_df[events_df['ICUSTAY_ID'].isin(valid_icustay_ids)].copy()
        self.logger.info(f"  Retained {len(valid_icustay_ids)} patients after filtering.")
        
        # ========== 2. Determine Outcome for Each Patient (Vectorized) ==========
        self.logger.info("  Determining patient outcomes...")
        pred_window_df = events_df[
            (events_df['hours_from_admission'] >= self.config.PREDICTION_WINDOW[0]) &
            (events_df['hours_from_admission'] <= self.config.PREDICTION_WINDOW[1])
        ]
        
        event_deterioration_ids = set()
        if not pred_window_df.empty:
            crit_lactate = set(pred_window_df.loc[(pred_window_df['item_name'] == 'lactate') & (pred_window_df['VALUENUM'] > 2.0), 'ICUSTAY_ID'])
            crit_sbp = set(pred_window_df.loc[(pred_window_df['item_name'] == 'sbp') & (pred_window_df['VALUENUM'] < 90), 'ICUSTAY_ID'])
            crit_resp = set(pred_window_df.loc[(pred_window_df['item_name'] == 'resp_rate') & (pred_window_df['VALUENUM'] > 30), 'ICUSTAY_ID'])
            crit_spo2 = set(pred_window_df.loc[(pred_window_df['item_name'] == 'spo2') & (pred_window_df['VALUENUM'] < 90), 'ICUSTAY_ID'])
            crit_creat = set(pred_window_df.loc[(pred_window_df['item_name'] == 'creatinine') & (pred_window_df['VALUENUM'] > 2.0), 'ICUSTAY_ID'])
            
            event_deterioration_ids = (crit_lactate & crit_sbp) | crit_resp | crit_spo2 | crit_creat

        mortality_df = self.cohort[self.cohort['ICUSTAY_ID'].isin(valid_icustay_ids)].copy()
        mortality_df['hours_to_death'] = (mortality_df['DEATHTIME'] - mortality_df['INTIME']).dt.total_seconds() / 3600
        mortality_ids = set(mortality_df.loc[
            (mortality_df['hours_to_death'] >= self.config.PREDICTION_WINDOW[0]) &
            (mortality_df['hours_to_death'] <= self.config.PREDICTION_WINDOW[1]), 'ICUSTAY_ID'
        ])

        deteriorated_ids = event_deterioration_ids | mortality_ids
        outcomes = pd.Series(0, index=valid_icustay_ids, name='deteriorated')
        outcomes.loc[list(deteriorated_ids)] = 1
        
        # ========== 3. Engineer Windowed Features (Vectorized) ==========
        self.logger.info("  Engineering windowed statistical features...")
        obs_events = events_df[events_df['hours_from_admission'] < 48].copy()
        window_bins = sorted(list(set([0] + [w[1] for w in self.config.OBSERVATION_WINDOWS])))
        window_labels = [f'window_{s}_{e}h' for s, e in self.config.OBSERVATION_WINDOWS]
        obs_events['window'] = pd.cut(obs_events['hours_from_admission'], bins=window_bins, labels=window_labels, right=False, include_lowest=True)
        
        stats = obs_events.groupby(['ICUSTAY_ID', 'window', 'item_name'])['VALUENUM'].agg(
            ['mean', 'std', 'min', 'max', 'median', 'count']
        )
        features_df = stats.unstack(level=['window', 'item_name'])
        features_df.columns = ['_'.join(col).strip() for col in features_df.columns.values]
        
        # ========== 4. Add Static Features, Clinical Scores, and Outcome ==========
        self.logger.info("  Adding static features and clinical scores...")
        static_features = self.cohort[self.cohort['ICUSTAY_ID'].isin(valid_icustay_ids)]
        static_features = static_features.drop_duplicates(subset=['ICUSTAY_ID']).set_index('ICUSTAY_ID')
        features_df['age'] = static_features['AGE']
        features_df['gender_male'] = (static_features['GENDER'] == 'M').astype(int)
        features_df['admission_type_emergency'] = (static_features['ADMISSION_TYPE'] == 'EMERGENCY').astype(int)
        
        features_df = self.feature_engineer._calculate_standard_clinical_scores(features_df)
        
        features_df = features_df.join(outcomes, how='left')
        features_df['deteriorated'] = features_df['deteriorated'].fillna(0).astype(int)

        self.features_df = features_df.reset_index()
        self.logger.info(f"Created features for {len(self.features_df)} patients")
        self.logger.info(f"Deterioration rate: {self.features_df['deteriorated'].mean():.2%}")
        self.features_df.to_csv(self.config.OUTPUT_PATH / 'features.csv', index=False)

    def _prepare_data(self) -> Tuple:
        """Prepare data for modeling."""
        X = self.features_df.drop(columns=['ICUSTAY_ID', 'deteriorated'])
        y = self.features_df['deteriorated'].values
        feature_names = X.columns.tolist()

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.VAL_SIZE / (1 - self.config.TEST_SIZE),
            random_state=self.config.RANDOM_STATE, stratify=y_temp
        )

        self.logger.info(f"Train set: {len(X_train)} ({y_train.mean():.2%} positive)")
        self.logger.info(f"Val set:   {len(X_val)} ({y_val.mean():.2%} positive)")
        self.logger.info(f"Test set:  {len(X_test)} ({y_test.mean():.2%} positive)")

        # ### MODIFICATION ### Using a much faster imputer.
        # For final model, you might switch back to KNNImputer after testing.
        self.logger.info("Imputing missing values with SimpleImputer (mean strategy)...")
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
        
        self.logger.info("Scaling features with RobustScaler...")
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        with open(self.config.OUTPUT_PATH / 'imputer.pkl', 'wb') as f: pickle.dump(imputer, f)
        with open(self.config.OUTPUT_PATH / 'scaler.pkl', 'wb') as f: pickle.dump(scaler, f)

        self.logger.info("Applying SMOTE to training data for class balance...")
        smote = SMOTE(random_state=self.config.RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        self.logger.info(f"After SMOTE: {len(X_train_balanced)} samples ({y_train_balanced.mean():.2%} positive)")

        return X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test, feature_names

    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Train and calibrate all models."""
        self.model_trainer.train_balanced_random_forest(X_train, y_train, X_val, y_val)
        self.model_trainer.train_xgboost(X_train, y_train, X_val, y_val)
        self.model_trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        self.model_trainer.train_lstm_attention(X_train, y_train, X_val, y_val)
        self.model_trainer.calibrate_models(X_val, y_val)
        self.models = self.model_trainer.models

    def _save_models_and_results(self, ensemble_weights: Dict, results: Dict):
        """Save trained models and results."""
        self.logger.info("\nSaving models and results...")
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(self.config.OUTPUT_PATH / f'{name}_model.h5')
            else:
                with open(self.config.OUTPUT_PATH / f'{name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        with open(self.config.OUTPUT_PATH / 'ensemble_weights.json', 'w') as f:
            json.dump(ensemble_weights, f, indent=2)

        with open(self.config.OUTPUT_PATH / 'evaluation_results.json', 'w') as f:
            results_serializable = {model_name: {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            } for model_name, metrics in results.items()}
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info("All models and results saved successfully!")

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Ensure the DATA_PATH in the Config class points to your MIMIC-III folder.
    # For example:
    # Config.DATA_PATH = Path('C:/Users/YourUser/mimic-data/mimic-iii-clinical-database-1.4')
    
    pipeline = ICUPipeline()
    pipeline.run()