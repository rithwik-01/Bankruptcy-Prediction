# Bankruptcy Prediction - Complete Project Guide

## **Project Overview**

ML-based techniques for handling class imbalance in bankruptcy prediction.
Follow-up to RGAN (deep learning) presentation, now exploring ML approaches.

---

## **Project Structure**

```
bankruptcy/
├── src/                          # Core utilities (reusable)
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── imbalance_techniques.py  # 9 imbalance handling methods
│   ├── models.py                # Model training & tuning
│   └── evaluation.py            # Metrics + threshold optimization
│
├── notebooks/                    # Analysis (run in order!)
│   ├── 01_imbalance_techniques_comparison.ipynb
│   ├── 02_xgboost_optimization.ipynb
│   └── 03_threshold_calibration.ipynb
│
├── api/                         # Flask API
│   ├── app.py                  # Main API
│   ├── models/                 # Saved models (.pkl files)
│   └── README.md               # API documentation
│
├── results/                     # CSV results
├── figures/                     # Visualizations
└── requirements.txt             # Dependencies
```

---

##  **Getting Started**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

**Notebook 01: Imbalance Techniques Comparison** (30-45 min)
- Compares 9 techniques: Baseline, Class Weights, SMOTE variants, etc.
- Tests with Logistic Regression + Random Forest
- Outputs: `results/imbalance_techniques_results.csv`, comparison charts

**Notebook 02: XGBoost Optimization** (15-20 min)
- XGBoost with `scale_pos_weight` for imbalance
- Randomized hyperparameter search (50 iterations)
- Feature importance analysis
- Outputs: `api/models/xgb_tuned.pkl`, `xgb_smote.pkl`

**Notebook 03: Threshold Optimization & Calibration** (10-15 min)
- **Threshold optimization** for F1, cost, Youden's J
- Model calibration (isotonic regression)
- ROC/PR curves, calibration curves
- Outputs: `api/models/xgb_calibrated.pkl`, `api/model_config.json`

---

## **Key Features Implemented**

### 1. Imbalance Handling (9 Techniques)
- Baseline (none)
- Class Weights
- Random Over/Under sampling
- SMOTE
- Borderline SMOTE
- ADASYN
- SMOTE-Tomek
- SMOTE-ENN

### 2. Models
- Logistic Regression
- Random Forest
- XGBoost (with tuning)
- Calibrated XGBoost

### 3. Evaluation Metrics
- Standard: Accuracy, Precision, Recall, F1
- **Imbalanced-specific: PR-AUC** (better than ROC-AUC)
- **Business: Cost calculation** (FP vs FN costs)
- **Threshold optimization** (not just 0.5!)

### 4. Threshold Optimization ⭐
- Optimize for F1-score
- Optimize for business cost
- Optimize for Youden's J statistic
- Visual analysis with 4-panel plots

### 5. Model Calibration
- Isotonic regression
- Calibration curves
- Reliable probability estimates

---

##  **Expected Results**

From preliminary testing:
- **Best technique:** SMOTE variants (Borderline SMOTE, ADASYN)
- **Best model:** XGBoost calibrated
- **Optimal threshold:** ~0.30-0.40 (NOT 0.5!)
- **F1-Score improvement:** ~40-60% over baseline
- **Recall:** 60-70% (catching most bankruptcies)

---

##  **Flask API Usage**

### Start API
```bash
cd api
python app.py
```

### Test Endpoints
```bash
# Check status
curl http://localhost:5000/

# Get models
curl http://localhost:5000/models

# Predict (example)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [/* 64 values */], "threshold": 0.35}'
```

---

##  **For Presentation**

### Key Points to Emphasize:

1. **ML Techniques** (after RGAN):
   - 9 different imbalance handling approaches
   - Comprehensive comparison with metrics

2. **Threshold Optimization** ⭐:
   - Default 0.5 is suboptimal
   - Business-driven threshold selection
   - Visual analysis tools

3. **Production-Ready**:
   - Calibrated probabilities
   - Flask API with multiple endpoints
   - Model performance monitoring

4. **Business Impact**:
   - Cost-based evaluation
   - Risk level classification
   - Interpretable results

### Figures to Show:
- `figures/01_imbalance_comparison.png` - Techniques comparison
- `figures/02_xgb_feature_importance.png` - Top features
- `figures/03_threshold_analysis.png` - Threshold optimization ⭐
- `figures/03_calibration_curves.png` - Model calibration

---

##  **Next Steps (If Time Permits)**

1. **SHAP Explainability** - Why predictions were made
2. **Time-based validation** - Use year data for temporal splits
3. **Ensemble methods** - Stack multiple models
4. **Docker deployment** - Containerize API

---

##  **For Teammates**

### To Reproduce Results:
1. Clone repo
2. `pip install -r requirements.txt`
3. Run notebooks 01 → 02 → 03 in order
4. Check `results/` and `figures/` folders
5. Start API: `cd api && python app.py`

### To Present:
1. Show project structure
2. Walk through Notebook 01 (techniques comparison)
3. Highlight Notebook 03 (threshold optimization) ⭐
4. Demo Flask API
5. Discuss business impact

---

##  **Checklist Before Presentation**

- [ ] All notebooks executed successfully
- [ ] Results saved in `results/` folder
- [ ] Figures generated in `figures/` folder
- [ ] API tested and working
- [ ] Best model identified
- [ ] Understand threshold optimization concept
- [ ] Know which technique performed best
- [ ] Can explain business cost calculation

---

**Questions?** Check individual READMEs in each folder.

**Good luck with your presentation!** 🚀

