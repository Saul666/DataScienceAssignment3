# REPORT — Module 3 · Assignment 3 · Deep Learning Foundations

**Name:** _IDAN GILAD__  **ID:** _038506432__  **Date:** _05/08/2026__
**Chosen option:** __A_ (A · Olist MLP / B · Fashion-MNIST CNN / C · Olist Autoencoder)

> Keep this report in English. A neural net that loses to a simpler model is a finding,
> not a failure. Report it honestly.

---

## 1. Framing
Task and primary metric:

Baseline you are comparing against (for A: your best Assignment-1 model):

The goal of Option A is to predict whether a customer will leave a negative review ($\text{review\_score} \le 2$) based on order, shipping, and payment features.Because negative reviews represent a minority class (~12–15% of total orders),  Accuracy is a misleading metric (a dummy classifier predicting "not negative" for everything achieves ~85% accuracy). 

Therefore, we evaluate models using:

ROC-AUC (Primary Metric): Measures ranking performance across all decision thresholds without being biased by class imbalance

.F1-Score (Secondary Metric): Balances Precision (avoiding false alarms) and Recall (catching actual unhappy customers) for the minority positive class ($\text{is\_negative} = 1$).

PR-AUC (Precision-Recall AUC): Tracks minority class performance directly under extreme class imbalance.

---

## 2. Results
| Model | Test metric | Params | Train time | Notes |
|---|---|---|---|---|
| simpler baseline | | | | |
| neural net | | | | |

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
Paste the completed DL Model Card from the notebook here.

```
(DL Model Card)
```

---

## 5. Reflection
What surprised you? When, in your mid-term project, would reaching for DL be the right call?
