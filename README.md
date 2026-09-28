# Predictive Maintenance: Machine Failure Prediction with Ensemble Learning

**Author:** Nicolette Mtisi  
**Tools:** Python · pandas · scikit-learn · imbalanced-learn · Matplotlib · Seaborn

## Overview
Unplanned equipment failures are expensive: they cause downtime, emergency repairs and lost production. This project predicts **whether a milling machine will fail** from its sensor readings. That way maintenance can be scheduled before a breakdown instead of after one.

**Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI), 10,000 synthetic records modeled on a real milling machine and included as `dataset.csv`. The features are air temperature, process temperature, rotational speed, torque, tool wear and product quality type (L/M/H). Only **3.4% of records are failures**, so the dataset is highly imbalanced.

## Approach
1. **EDA:** distributions by product type, the torque-vs-speed relationship coloured by failure, and a correlation analysis. Air and process temperature are strongly correlated, and so are torque and speed.
2. **Preprocessing:** dropped identifiers and the individual failure-mode flags (TWF, HDF, PWF, OSF, RNF) to avoid **target leakage**, then ranked features.
3. **Modeling:** six classifiers trained on an 80/20 split:
   Support Vector Machine · Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · MLP Neural Network
4. **Stacking ensemble:** combines all six base models' predictions through a meta-learner.
5. **Evaluation:** accuracy plus **precision, recall and F1 for the failure class**, confusion matrices, and k-fold cross-validation.

## Results
With only 3.4% failures, a model that *always* predicts "no failure" would already be about 97% accurate. So the **failure-class metrics** are what really matter here.

| Model | Accuracy | Failure Precision | Failure Recall | Failure F1 |
|-------|---------:|------------------:|---------------:|-----------:|
| SVM | 0.97 | 1.00 | 0.03 | 0.06 |
| Logistic Regression | 0.97 | 0.62 | 0.26 | 0.37 |
| MLP Neural Network | 0.98 | 0.77 | 0.39 | 0.52 |
| Random Forest | 0.98 | 0.80 | 0.59 | 0.68 |
| Decision Tree | 0.98 | 0.66 | **0.72** | 0.69 |
| Gradient Boosting | 0.98 | 0.84 | 0.59 | 0.69 |
| **Stacking Ensemble** ⭐ | **0.985** | **0.88** | 0.59 | **0.71** |

**Key takeaways**
- The **stacking ensemble** had the best overall balance: 98.5% accuracy (98.4–98.6% across cross-validation runs) and the highest failure F1. When it flags a failure, it's right **88%** of the time.
- The **tree-based models** clearly beat the linear models and SVM, which missed almost every failure.
- The **Decision Tree** caught the most failures (72% recall). That's worth considering where a missed failure costs more than a false alarm.

## Next Steps
- **Train on the rebalanced data.** The notebook prepares SMOTE oversampling plus undersampling (`X_train_resampled`), but the models above were trained on the original, imbalanced split. Training on the balanced set, or using class weights, should raise failure recall.
- Tune the decision threshold to trade precision for recall, depending on maintenance costs.
- Engineer physics-based features such as **power** (torque × rotational speed) and the **temperature difference**.

## How to Run
```bash
git clone https://github.com/nic-stack/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
jupyter notebook predictive_maintenance_models.ipynb
```

## Files
| File | Description |
|------|-------------|
| `predictive_maintenance_models.ipynb` | EDA, preprocessing, the six models and the stacking ensemble |
| `predictive_maintenance_evaluation.ipynb` | Extended evaluation, including cross-validation of the stacking model |
| `dataset.csv` | AI4I 2020 Predictive Maintenance dataset |
| `requirements.txt` | Python dependencies |

## Links
- 📝 [Medium write-up](https://medium.com/@nicmtisi/enhancing-predictive-maintenance-through-machine-learning-a-comprehensive-survey-on-ensemble-d6dc0ed5898c)
- 🌐 [Portfolio](https://nic-stack.github.io/NicoletteMtisi/) · [LinkedIn](https://www.linkedin.com/in/nicolette-mtisi)
