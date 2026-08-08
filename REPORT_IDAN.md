# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:** _IDAN GILAD__  **ID:** _038506432__  **Date:** _05/08/2026__
**Chosen option:** __A_ (A · Olist MLP / B · Fashion-MNIST CNN / C · Olist Autoencoder)



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

### Did DL win?
Yes, the PyTorch MLP won decisively against the Gradient Boosting baseline. It achieved a ROC-AUC of 0.7485 versus 0.6800 (+0.0685 lift) and an F1-Score of 0.3345 versus 0.2050 (+0.1198 lift). Most importantly, minority class recall nearly tripled from 12.1% to 33.1%, catching significantly more dissatisfied customers at the cost of a minor drop in interpretability.

### Logits / loss
`nn.BCEWithLogitsLoss()` was used because it combines a Sigmoid activation and Binary Cross-Entropy loss into a single class. It uses the log-sum-exp trick for numerical stability to prevent floating-point underflow or overflow during backpropagation. If `torch.sigmoid()` or `softmax` is applied before passing predictions into this loss, the sigmoid operation gets applied twice, distorting the gradients and leading to severe numerical instability.

### Overfitting
The training and validation loss curves did not diverge, both converging smoothly around Epoch 5 (~0.322 train vs ~0.319 val) and remaining flat through Epoch 20. Validation loss remained slightly lower than training loss because `Dropout(0.2)` was active during training but disabled during evaluation. Because the loss curves remained aligned and flat, no additional anti-overfitting measures were needed.

### Learning rate
At a too-large learning rate (`1e-1`), training was unstable and loss oscillated wildly because gradient steps overshot local minima. At a too-small learning rate (`1e-5`), the loss decreased sluggishly and failed to converge within 20 epochs. Learning rate is the most critical hyperparameter because it governs optimization step size—dictating whether a model converges, stalls, or diverges.

### Regularization
`Dropout(p=0.2)` was applied after the 32-unit hidden layer. It successfully prevented overfitting on tabular features, keeping validation loss stable (~0.319) alongside training loss (~0.322) across all 20 epochs. Without dropout, small tabular networks risk memorizing noise in minority class samples.

### Cost / benefit
The MLP uses only 321 trainable parameters and trains in ~2 seconds (20 epochs) compared to ~1 second for Gradient Boosting. While tree models offer easier direct feature importance inspection, the MLP adds almost zero compute overhead. The +0.0685 ROC-AUC gain and 3x recall boost fully justify the slight loss of tree-style interpretability.

### When DL
I would deploy the PyTorch MLP for this task. The +0.0685 ROC-AUC lift and the 3x increase in detecting dissatisfied customers (33.1% recall vs 12.1%) directly impact customer retention and business value. Because the network contains only 321 parameters, CPU inference latency remains sub-millisecond, removing any operational performance penalty.

### Monday morning
In production, I would monitor feature drift on critical delivery metrics like `delay_days` (using Population Stability Index or KS-tests), output probability score distributions, and rolling test ROC-AUC scores. A model retrain would be triggered on a monthly schedule, or automatically if Population Stability Index exceeds 0.25 or rolling ROC-AUC drops below 0.70.

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

## Reflection

### What surprised you?
What surprised me most was how significantly a lightweight, 321-parameter MLP outperformed classical ensemble models like Gradient Boosting on tabular data. Standard intuition often favors tree-based models for small tabular datasets, but proper optimization using `BCEWithLogitsLoss` and `Dropout(0.2)` allowed the neural network to nearly triple minority class recall (from 12.1% to 33.1%) while maintaining rapid 2-second convergence. Additionally, observing the extreme sensitivity of training stability to learning rate—where shifting by a factor of 10 caused total instability (`1e-1`) or underfitting (`1e-5`)—highlighted just how crucial hyperparameter selection is for deep learning.

### When, in your pair trading mid-term project, would reaching for DL be the right call?
Reaching for Deep Learning in pair trading is justified when traditional linear assumptions (such as static linear cointegration via the Engle-Granger test) break down due to market non-linearities and regime shifts. DL becomes the right call when:
1. **High-Dimensional / Alternative Data Integration:** Modeling complex, non-linear relationships across hundreds of candidate pairs simultaneously while incorporating order book depth (LOB), sentiment scores, and macroeconomic time series.
2. **Dynamic Spread & Regime Modeling:** Using sequence architectures (like LSTMs, TCNs, or Temporal Fusion Transformers) to predict time-varying mean-reversion speed, dynamic hedge ratios, and structural breaks in time-series spreads.
3. **Execution & Position Sizing:** Applying Deep Reinforcement Learning (DRL) to dynamically optimize entry/exit thresholds, trade execution timing, and position sizing while accounting for transaction costs and market impact.
