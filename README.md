# Bankruptcy Prediction using ML Techniques for Imbalanced Data

A comprehensive machine learning project for predicting company bankruptcy, focusing on handling severe class imbalance through various resampling techniques, model optimization, and threshold calibration.

---

## 📋 Project Overview

This project explores **ML-based approaches** for bankruptcy prediction as a follow-up to RGAN (deep learning) research. The primary challenge addressed is **severe class imbalance** (95% non-bankruptcy vs 5% bankruptcy cases).

### Key Objectives
- Compare 9 different imbalance handling techniques
- Optimize XGBoost with hyperparameter tuning
- Implement threshold optimization for business-driven decisions
- Deliver a production-ready Flask API

---

## 📊 Dataset

**Polish Companies Bankruptcy Dataset**
- **Total samples:** 43,405 companies
- **Features:** 64 financial ratios
- **Class distribution:** 
  - Non-bankrupt: 41,314 (95.2%)
  - Bankrupt: 2,091 (4.8%)
- **Imbalance ratio:** ~20:1

The dataset spans 5 years of financial data from Polish companies.

---

## 🔬 Methodology & Results

### Phase 1: Imbalance Techniques Comparison

Compared **9 techniques** with Logistic Regression and Random Forest:

| Technique | Model | F1-Score | Recall | PR-AUC |
|-----------|-------|----------|--------|--------|
| Random Oversampling | Random Forest | **0.3203** | 0.5767 | 0.2706 |
| Class Weights | Random Forest | 0.3068 | 0.4569 | 0.2511 |
| SMOTE | Random Forest | 0.2908 | 0.6459 | 0.2641 |
| ADASYN | Random Forest | 0.2906 | 0.6651 | 0.2742 |
| Borderline SMOTE | Random Forest | 0.2871 | 0.5837 | 0.2328 |
| Random Undersampling | Random Forest | 0.2299 | **0.7727** | 0.2137 |
| Baseline (None) | Random Forest | 0.0505 | 0.0263 | 0.3362 |

**Key Finding:** SMOTE variants and oversampling significantly outperform baseline, with Random Forest consistently beating Logistic Regression.

### Phase 2: XGBoost Optimization

Performed **randomized hyperparameter search** (50 iterations, 5-fold CV):

| Configuration | F1-Score | Precision | Recall | PR-AUC | Total Cost |
|---------------|----------|-----------|--------|--------|------------|
| XGBoost Baseline | 0.4638 | 0.3420 | 0.7201 | 0.5676 | $14.6M |
| **XGBoost Tuned** | **0.5826** | **0.5792** | 0.5861 | **0.6200** | $18.2M |
| XGBoost + SMOTE | 0.4051 | 0.2707 | **0.8038** | 0.5806 | $12.7M |

**Best Parameters Found:**
```
n_estimators: 500
max_depth: 7
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.9
min_child_weight: 5
scale_pos_weight: 29.63
```

### Phase 3: Threshold Optimization & Calibration

**Critical insight:** Default threshold (0.5) is suboptimal!

| Threshold | Precision | Recall | F1-Score | Total Cost |
|-----------|-----------|--------|----------|------------|
| Default (0.50) | 0.5792 | 0.5861 | 0.5826 | $18.2M |
| Optimal F1 (0.55) | **0.6290** | 0.5598 | **0.5924** | $19.1M |
| Optimal Cost (0.17) | 0.3194 | **0.8254** | 0.4606 | **$11.0M** |
| Conservative (0.30) | 0.4339 | 0.6986 | 0.5353 | $14.5M |

**Final Production Model:** Calibrated XGBoost with isotonic regression
- Threshold optimized for business cost (0.17)
- Catches 82.5% of bankruptcies
- Minimizes total business cost

---

## 📈 Key Improvements Achieved

| Metric | Baseline | Best Model | Improvement |
|--------|----------|------------|-------------|
| F1-Score | 0.02 | 0.59 | **+2850%** |
| Recall | 0.01 | 0.83 | **+8200%** |
| PR-AUC | 0.20 | 0.62 | **+210%** |

---

## 🗂️ Project Structure

```
Bankruptcy-Prediction/
├── notebooks/
│   ├── 01_imbalance_techniques_comparison.ipynb  # Compare 9 techniques
│   ├── 02_xgboost_optimization.ipynb             # XGBoost tuning
│   └── 03_threshold_calibration.ipynb            # Threshold optimization
├── src/                                          # Core utilities
│   ├── data_loader.py                            # Data loading & preprocessing
│   ├── imbalance_techniques.py                   # 9 imbalance methods
│   ├── models.py                                 # Model training & tuning
│   └── evaluation.py                             # Metrics & threshold optimization
├── api/                                          # Flask API
│   ├── app.py                                    # REST API endpoints
│   ├── models/                                   # Saved model files (.pkl)
│   └── model_config.json                         # Production configuration
├── results/                                      # CSV results
├── figures/                                      # Visualizations
├── requirements.txt                              # Dependencies
├── EXECUTION_ORDER.md                            # Step-by-step guide
└── PROJECT_GUIDE.md                              # Detailed documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks (in order)

```bash
# Notebook 1: Compare imbalance techniques (30-45 min)
jupyter notebook notebooks/01_imbalance_techniques_comparison.ipynb

# Notebook 2: XGBoost optimization (15-20 min)
jupyter notebook notebooks/02_xgboost_optimization.ipynb

# Notebook 3: Threshold calibration (10-15 min)
jupyter notebook notebooks/03_threshold_calibration.ipynb
```

### 3. Start Flask API

```bash
cd api
python app.py
```

### 4. Test API Endpoints

```bash
# Check status
curl http://localhost:5000/

# List available models
curl http://localhost:5000/models

# Get model performance
curl http://localhost:5000/model_performance

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [/* 64 feature values */], "threshold": 0.17}'
```

---

## 📦 Dependencies

```
numpy>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.14.0
xgboost>=3.0.0
matplotlib>=3.10.0
seaborn>=0.11.0
shap>=0.50.0
flask>=2.0.0
flask-cors>=3.0.10
jupyter>=1.0.0
```

---

## 🎯 Techniques Implemented

### Imbalance Handling (9 Methods)
1. **Baseline** - No handling
2. **Class Weights** - Penalize misclassification of minority class
3. **Random Oversampling** - Duplicate minority samples
4. **Random Undersampling** - Remove majority samples
5. **SMOTE** - Synthetic Minority Oversampling
6. **Borderline SMOTE** - Focus on decision boundary samples
7. **ADASYN** - Adaptive synthetic sampling
8. **SMOTE-Tomek** - SMOTE + Tomek links cleaning
9. **SMOTE-ENN** - SMOTE + Edited Nearest Neighbors

### Models
- Logistic Regression (with class weights)
- Random Forest (with class weights)
- XGBoost (with `scale_pos_weight`)
- Calibrated XGBoost (isotonic regression)

### Evaluation Metrics
- **Standard:** Accuracy, Precision, Recall, F1-Score
- **Imbalanced-specific:** PR-AUC (Precision-Recall AUC)
- **Business:** Cost calculation (FP: $5,000, FN: $100,000)
- **Threshold optimization:** F1, Cost, Youden's J

---

## 💡 Key Insights

1. **SMOTE variants** provide the best balance between precision and recall
2. **Random Forest** consistently outperforms Logistic Regression
3. **XGBoost with tuning** achieves the highest F1-score (0.58)
4. **Threshold optimization is critical** - default 0.5 is suboptimal
5. **Business cost optimization** (threshold 0.17) catches 82.5% of bankruptcies
6. **Model calibration** improves probability reliability for risk assessment

---

## 📊 Output Files

After running all notebooks:

| File | Description |
|------|-------------|
| `results/imbalance_techniques_results.csv` | Comparison of 9 techniques |
| `results/xgboost_results.csv` | XGBoost configurations comparison |
| `figures/01_imbalance_comparison.png` | Techniques comparison chart |
| `figures/01_metrics_heatmap.png` | Metrics heatmap by technique |
| `figures/02_xgb_feature_importance.png` | Top 20 important features |
| `figures/03_threshold_analysis.png` | Threshold optimization plots |
| `figures/03_calibration_curves.png` | Model calibration curves |
| `api/models/xgb_tuned.pkl` | Tuned XGBoost model |
| `api/models/xgb_smote.pkl` | XGBoost + SMOTE model |
| `api/models/xgb_calibrated.pkl` | **Production model** |
| `api/model_config.json` | Production configuration |

---

## 🔮 Future Improvements

- [ ] SHAP explainability for model interpretability
- [ ] Time-based validation using year data
- [ ] Ensemble stacking of multiple models
- [ ] Docker containerization for deployment
- [ ] Real-time monitoring dashboard

---

## 📚 References

- Polish Companies Bankruptcy Dataset (UCI ML Repository)
- SMOTE: Chawla et al. (2002)
- XGBoost: Chen & Guestrin (2016)
- Imbalanced-learn library documentation

---

## 👥 Contributors

This project was developed as part of bankruptcy prediction research, exploring ML alternatives to deep learning (RGAN) approaches.

---

## 📄 License

This project is for educational and research purposes.

