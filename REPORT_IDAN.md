# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:** _IDAN GILAD__  **ID:** _038506432__  **Date:** _05/08/2026__
**Chosen option:** __A_ (A · Olist MLP / B · Fashion-MNIST CNN / C · Olist Autoencoder)

> Keep this report in English. A neural net that loses to a simpler model is a finding,

---

## 1. Framing
Task and primary metric:

Baseline you are comparing against (for A: your best Assignment-1 model):

The goal of Option A is to predict whether a customer will leave a negative review based on order, shipping, and payment features.Because negative reviews represent a minority class,  Accuracy is a misleading metric (a dummy classifier predicting "not negative" for everything achieves ~85% accuracy). 

Therefore, we evaluate models using:

ROC-AUC (Primary Metric): Measures ranking performance across all decision thresholds without being biased by class imbalance

.F1-Score (Secondary Metric): Balances Precision (avoiding false alarms) and Recall (catching actual unhappy customers) for the minority positive class .

PR-AUC (Precision-Recall AUC): Tracks minority class performance directly under extreme class imbalance.

---

## 2. Results

| Model | Test metric | Params | Train time | Notes |
|---|---|---|---|---|
| Simpler baseline (Gradient Boosting) | ROC-AUC: 0.6800<br>F1-Score: 0.2050 | Non-parametric (Tree Ensemble) | ~1.0s (CPU) | High precision (67.9%), but missed ~88% of negative reviews due to low recall (12.1%). |
| Neural Net (PyTorch MLP) | **ROC-AUC: 0.7485**<br>**F1-Score: 0.3345** | 321 | ~2.0s (20 epochs) | Outperformed baseline (+0.0685 ROC-AUC lift); nearly 3x higher recall (33.1%); stable convergence by Epoch 5. |

---

## 3. Guiding questions (graded)
Answer each in 2-5 sentences.

1. **Did DL win?** Did your neural net beat the simpler / cheaper baseline? By how much, and at what cost?
2. **Logits / loss.** Which loss did you use and why? What breaks if you apply softmax / sigmoid before it?
3. **Overfitting.** Read your learning curves. Where do train and val diverge, and what did you do about it?
4. **Learning rate.** Show what happens with a too-large and a too-small LR. Why is it the most important knob?
5. **Regularization.** Which regularizer did you use, and did it actually help (with numbers)?
6. **Cost / benefit.** Count the cost of the NN (params, training time, interpretability) vs the simple model. Is it justified?
7. **When DL.** For your task, would you deploy DL or the simpler model? Defend it.
8. **Monday morning.** What would you monitor in production, and what would trigger a retrain?

---

## 4. DL Model Card

# DL Model Card

## 1. Overview
- Option / task / data: Option A · Olist Tabular Negative-Review Binary Classification (predicting `is_negative = 1` for review scores <= 2 using order, logistics, and payment features).
- Architecture (layers, params): Multi-Layer Perceptron (MLP) with 8 input features -> Linear(8, 32) -> ReLU -> Dropout(0.2) -> Linear(32, 1). Total trainable parameters: 321.

## 2. Setup
- Loss and why (logits handling): `nn.BCEWithLogitsLoss()`. It combines a Sigmoid layer and Binary Cross-Entropy in a single class using the log-sum-exp trick for numerical stability, preventing underflow/overflow. Output layer outputs raw logits.
- Optimizer, learning rate, regularizer: Adam optimizer (`lr=0.001`), `Dropout(p=0.2)` on the hidden layer for regularization, default L2 weight decay (0.0).

## 3. Performance
- Simpler-model baseline: Gradient Boosting (Assignment 1) | ROC-AUC: 0.6800, F1-Score: 0.2050, Recall: 0.1208.
- Neural-net test metric: PyTorch MLP | ROC-AUC: 0.7485, F1-Score: 0.3345, Precision: 0.3380, Recall: 0.3310.
- Did DL win? By how much, at what cost? Yes, DL won significantly. ROC-AUC improved by +0.0685 and F1-Score by +0.1198. Minority class recall nearly tripled (12% -> 33%). Cost: Negligible compute overhead (321 parameters), but slight reduction in model interpretability compared to decision trees.

## 4. Diagnostics
- Learning curves: Train and val loss converge around Epoch 5 (~0.322 train loss vs ~0.319 val loss) and remain flat through Epoch 20. The curves do not diverge, indicating no severe overfitting thanks to Dropout regularization.
- Learning-rate sensitivity: 
  * `lr=1e-1`: Unstable training; gradient steps overshoot the local minimum, causing loss oscillation.
  * `lr=1e-3`: Optimal learning rate; quick convergence in ~5 epochs with stable validation loss.
  * `lr=1e-5`: Too slow; loss decreases sluggishly and underfits within 20 epochs.

## 5. Decision
- Would you deploy DL or the simpler model here? Defend it. Deploy the PyTorch MLP. The +0.0685 ROC-AUC lift and 3x boost in capturing dissatisfied customers (Recall) provide strong business value for customer retention. With only 321 parameters, inference latency is near-instantaneous (<1ms on CPU), making the compute cost negligible.
- Production: what to monitor, what triggers a retrain.
  * Monitor: Data drift on critical delivery metrics (`delay_days`, `delivery_days` via Population Stability Index / KS-test), output score probability distribution drift, and rolling test F1/ROC-AUC scores.
  * Retrain Triggers: Monthly scheduled retrain on fresh order data, or immediate automated retrain if feature drift exceeds threshold (PSI > 0.25) or rolling ROC-AUC drops below 0.70.


---

## 5. Reflection
What surprised you? When, in your mid-term project, would reaching for DL be the right call?
