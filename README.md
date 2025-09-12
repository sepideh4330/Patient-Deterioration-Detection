# Download dataset from https://drive.google.com/file/d/1WToNusanHToSsB1505g4qaHhcbgc-PRb/view
# ICU Deterioration Prediction — Early Warning System (EWS)

End-to-end pipeline and demo app to predict ICU patient deterioration using early ICU data (0–48h) while **preventing temporal leakage**. Includes a training pipeline (`main.py`), saved models + preprocessors, and a Streamlit web app (`web_app.py`) for bedside-style exploration.

> **Scope:** Research prototype for academic evaluation. **Not** a medical device. Do **not** use for clinical decision-making.

---

## ✨ Highlights

- **Leakage-aware temporal framing**: Observation 0–48h → Gap 48–50h → Predict 50–74h.
- **Feature set**: Windowed statistics (mean/std/min/max/median/count) of vitals/labs/GCS/urine + SIRS & qSOFA.
- **Models**: Balanced Random Forest, XGBoost, LightGBM, (optional) LSTM; **Ensemble** via weighted average `{'random_forest': 0.25, 'xgboost': 0.25, 'lightgbm': 0.25, 'lstm': 0.25}`.
- **Metrics tracked**: AUC-ROC, AUC-PR, F1, Accuracy, Brier, MCC, Cohen’s Kappa, Sensitivity/Specificity, PPV/NPV.
- **Web demo**: Streamlit app with per-model + ensemble predictions, and simple clinical rules (SIRS/qSOFA).

---

## 📂 Repository Structure (suggested)

```
.
├── main.py                  # Training & evaluation pipeline (edit Config.DATA_PATH)
├── web_app.py               # Streamlit demo (expects artifacts nearby or in ./output)
├── output/                  # Generated artifacts (after training)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── lstm_model.h5              # optional
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── ensemble_weights.json
│   └── evaluation_results.json
├── cache/                   # Intermediate caches (optional)
├── data/                    # (not tracked) MIMIC-III v1.4 CSVs
└── README.md
```

> **Data note:** MIMIC-III requires credentialed access. Do not commit patient data.

---

## 🧰 Environment Setup

**Python**: 3.10–3.11 recommended.

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install core dependencies
pip install --upgrade pip wheel
pip install pandas numpy scikit-learn imbalanced-learn lightgbm xgboost tqdm joblib streamlit matplotlib

# Optional (for LSTM)
pip install tensorflow  # or tensorflow-cpu
```
> If LightGBM fails to install on your platform, consult the official install notes (may require build tools).

---

## ⚙️ Configuration

Open **`main.py`** and set the paths in the `Config` section:

```python
class Config:
    DATA_PATH = Path('./mimic-iii-clinical-database-1.4')  # <-- point to your MIMIC-III folder
    OUTPUT_PATH = Path('./output')
    CACHE_PATH = Path('./cache')
    # windows, random seeds, etc...
```

Key parameters you can tune:
- `OBSERVATION_WINDOWS`, `GAP_WINDOW`, `PREDICTION_WINDOW`
- `CHUNK_SIZE`, `N_JOBS`, `RANDOM_STATE`
- Model hyperparameters inside the training section

---

## 🚀 Training & Evaluation

```bash
# run the pipeline (after editing Config paths)
python main.py
```

Artifacts written to `OUTPUT_PATH`:
- `*_model.pkl` / `lstm_model.h5` (trained models)
- `imputer.pkl`, `scaler.pkl`
- `ensemble_weights.json`
- `evaluation_results.json` (test-set metrics)

---

## 📊 Results (from current repo artifacts)

| Model | AUC_ROC | AUC_PR | F1 | ACCURACY | BRIER | MCC | KAPPA |
|---|---|---|---|---|---|---|---|
| ENSEMBLE | 0.808 | 0.841 | 0.717 | 0.724 | 0.1800 | 0.450 | 0.449 |
| LIGHTGBM | 0.808 | 0.828 | 0.719 | 0.723 | 0.1757 | 0.446 | 0.446 |
| LSTM | 0.553 | 0.566 | 0.365 | 0.540 | 0.2477 | 0.104 | 0.086 |
| RANDOM_FOREST | 0.800 | 0.816 | 0.717 | 0.724 | 0.1789 | 0.449 | 0.448 |
| XGBOOST | 0.805 | 0.824 | 0.726 | 0.722 | 0.1773 | 0.444 | 0.444 |

> **Tip:** AUC-PR is often more telling under class imbalance. Consider reporting both AUC-ROC and AUC-PR.

---

## 🖥️ Web App (Streamlit)

Run the demo UI:
```bash
# Option A: copy artifacts next to web_app.py (default behavior)
cp output/*.pkl output/*.json output/*.h5 .  # copy what exists
streamlit run web_app.py

# Option B: point the app to ./output (edit the app to use Path(__file__).parent / "output")
# in web_app.py, set:
#   self.model_path = Path(__file__).parent / "output"
```
The app loads: `random_forest_model.pkl`, `xgboost_model.pkl`, `lightgbm_model.pkl`, `imputer.pkl`, `scaler.pkl`, and `ensemble_weights.json`. It computes SIRS/qSOFA, assembles features to match training, shows per-model and ensemble risk, and suggests a basic action message (for demo only).

---

## 🧪 Reproducibility Checklist

- Fix your **DATA_PATH** and ensure MIMIC-III tables are accessible.
- Verify temporal windows: **Obs 0–48h → Gap 48–50h → Predict 50–74h**.
- Re-run training with a fixed `RANDOM_STATE`.
- Export metrics via `evaluation_results.json` and compare with this README’s table.
- Optional: add **calibration** (Platt/Isotonic) and **decision curves** to select thresholds.

---

## 🧭 Roadmap

- Lead-time gain quantification (needs event onset time).
- Calibration (isotonic/Platt) + threshold policy per use-case.
- Fairness checks (subgroup AUC/PPV) and external validation (eICU/local).
- Harden demo into a containerized REST service with auth, audit, monitoring.

---

## 🙏 Acknowledgements

- MIMIC-III v1.4 by PhysioNet/Computing in Cardiology Challenge, Beth Israel Deaconess Medical Center & MIT.
- Open-source libraries: pandas, numpy, scikit-learn, imbalanced-learn, lightgbm, xgboost, tensorflow, streamlit.

---

## ⚠️ Disclaimer

This repository is for **research and educational purposes only**. It is **not** a medical device and must **not** be used to make clinical decisions.

---

## 📜 License

Add your license of choice (e.g., MIT/Apache-2.0/GPL-3.0) and a `LICENSE` file.

---

## 📬 Contact

For questions, please open an issue or contact the author/maintainer.
