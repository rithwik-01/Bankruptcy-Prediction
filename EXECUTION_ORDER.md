## What's Been Created

### Core Utilities (`src/` folder):
- `data_loader.py` - Loads & preprocesses bankruptcy data
- `imbalance_techniques.py` - 9 imbalance handling methods
- `models.py` - Model training & hyperparameter tuning
- `evaluation.py` - **Threshold optimization** + comprehensive metrics

### Notebooks (Ready to Execute):
- `01_imbalance_techniques_comparison.ipynb` - Compare 9 techniques
- `02_xgboost_optimization.ipynb` - XGBoost tuning
- `03_threshold_calibration.ipynb` - Threshold optimization ⭐

### Flask API:
- `api/app.py` - Full REST API for predictions
- `api/README.md` - API documentation

---

## 📝 **Step-by-Step Execution**

### **Step 1: Install Dependencies** (2 minutes)

```bash
cd /Users/ritwikreddy/Desktop/bankruptcy
pip install -r requirements.txt
```

### **Step 2: Run Notebook 01** (30-45 minutes) 

**What it does:**
- Compares 9 imbalance handling techniques
- Trains Logistic Regression + Random Forest with each
- Generates comparison charts

**Execute:**
```bash
jupyter notebook notebooks/01_imbalance_techniques_comparison.ipynb
```

**Outputs:**
- `results/imbalance_techniques_results.csv`
- `figures/01_imbalance_comparison.png`
- `figures/01_metrics_heatmap.png`

**Expected Best Results:**
- SMOTE, Borderline SMOTE, or ADASYN
- Random Forest > Logistic Regression
- F1-Score: ~0.25-0.35 (up from ~0.02 baseline!)

---

### **Step 3: Run Notebook 02** (15-20 minutes) 

**What it does:**
- XGBoost with `scale_pos_weight` for imbalance
- Randomized hyperparameter search (50 iterations)
- Tests XGBoost + SMOTE combination
- Feature importance analysis

**Execute:**
```bash
jupyter notebook notebooks/02_xgboost_optimization.ipynb
```

**Outputs:**
- `api/models/xgb_tuned.pkl` ← Best XGBoost model
- `api/models/xgb_smote.pkl` ← XGBoost + SMOTE
- `results/xgboost_results.csv`
- `figures/02_xgb_feature_importance.png`

**Expected:**
- XGBoost outperforms RF
- F1-Score: ~0.30-0.40
- PR-AUC: ~0.40-0.50

---

### **Step 3: Run Notebook 03** (10-15 minutes) 

**What it does:**
- **Threshold optimization** (your favorite!)
- Find optimal thresholds for F1, cost, Youden's J
- Model calibration (isotonic regression)
- Comprehensive visual analysis

**Execute:**
```bash
jupyter notebook notebooks/03_threshold_calibration.ipynb
```

**Outputs:**
- `api/models/xgb_calibrated.pkl` ← **PRODUCTION MODEL**
- `api/model_config.json` ← Configuration for API
- `figures/03_threshold_analysis.png` 
- `figures/03_calibration_curves.png`

**Key Insight:**
- Default threshold (0.5) is suboptimal!
- Optimal threshold: ~0.30-0.40 (based on business costs)
- Calibration improves probability estimates

---

### **Step 4: Test Flask API** (5 minutes)

**Start API:**
```bash
cd api
python app.py
```

**Test Endpoints:**

```bash
# 1. Check status
curl http://localhost:5000/

# 2. List models
curl http://localhost:5000/models

# 3. Get performance metrics
curl http://localhost:5000/model_performance

# 4. Test prediction (you'll need actual feature values)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [/* your 64 feature values */], "threshold": 0.35}'
```

---

##  **What to Show in Presentation**

### 1. Problem Statement
- Severe class imbalance: 95% non-bankruptcy, 5% bankruptcy
- After RGAN (deep learning), exploring ML techniques

### 2. Notebook 01 Highlights
- Show comparison of 9 techniques
- Display heatmap of results
- **Key point:** SMOTE variants win

### 3. Notebook 02 Highlights
- XGBoost performance
- Feature importance chart
- **Key point:** XGBoost > Random Forest

### 4. Notebook 03 Highlights ⭐
- **Threshold optimization plots** (4-panel visualization)
- Show how optimal threshold ≠ 0.5
- Calibration curves
- **Key point:** Business-driven threshold selection

### 5. Flask API Demo
- Show `/predict` endpoint working
- Explain risk levels
- **Key point:** Production-ready deployment

---

##  **Expected Timeline**

| Step | Time | Cumulative |
|------|------|------------|
| Install dependencies | 2 min | 2 min |
| Notebook 01 | 30-45 min | ~45 min |
| Notebook 02 | 15-20 min | ~65 min |
| Notebook 03 | 10-15 min | ~80 min |
| Test API | 5 min | ~85 min |

**Total: ~1.5 hours to complete everything**

---

##  **Key Metrics to Track**

After running all notebooks, you should have:

| Metric | Baseline | Best Technique | Improvement |
|--------|----------|----------------|-------------|
| F1-Score | ~0.02 | ~0.30-0.40 | +1500% |
| Recall | ~0.01 | ~0.60-0.70 | +6000% |
| PR-AUC | ~0.20 | ~0.40-0.50 | +100% |

---

##  **Common Issues & Solutions**

### Issue: Module not found error
**Solution:** Make sure you're in the right directory and run:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/ritwikreddy/Desktop/bankruptcy"
```

### Issue: XGBoost tuning takes too long
**Solution:** Reduce `n_iter` in Notebook 02 from 50 to 20

### Issue: Notebook cells fail
**Solution:** Run cells in order! Some cells depend on previous results

### Issue: API can't find models
**Solution:** Make sure you ran Notebooks 02 and 03 first to generate models

---

##  **For Teammates**

Share these files:
1. This `EXECUTION_ORDER.md` 
2. `PROJECT_GUIDE.md` - Overall structure
3. `requirements.txt` - Dependencies
4. Entire `src/` folder - Core utilities
5. All notebooks in `notebooks/`
6. `api/` folder - Flask API

They just need to:
1. Install dependencies
2. Run notebooks in order
3. Check `results/` and `figures/`

---

##  **Presentation Tips**

### Opening (2 min):
- "After RGAN, we explored 9 ML techniques for imbalance"
- Show project structure

### Main Demo (8 min):
1. **Techniques Comparison** - Show heatmap from Notebook 01
2. **XGBoost Results** - Show feature importance from Notebook 02
3. **Threshold Optimization** ⭐ - Show 4-panel plot from Notebook 03
4. **Live API Demo** - Quick prediction call

### Closing (2 min):
- Best technique: SMOTE + XGBoost
- Key insight: Threshold optimization matters!
- Production-ready API available

---

##  **Final Checklist**

Before presentation:
- [ ] All notebooks executed
- [ ] All figures generated in `figures/`
- [ ] Models saved in `api/models/`
- [ ] API tested and working
- [ ] Can explain threshold optimization
- [ ] Know which technique won
- [ ] Understand business cost calculation

---

**Ready to go! Start with Step 1 above.** 🚀

Questions? Check `PROJECT_GUIDE.md` for more details.

