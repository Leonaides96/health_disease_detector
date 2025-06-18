# 🧠 Health Disease Detector – Evaluation Metrics Guide

This project uses a binary classifier to detect whether a person has a particular disease. To evaluate the effectiveness of the model, we focus on three key performance metrics:

---

## 📊 1. Recall (Sensitivity / True Positive Rate)

**Formula:**

\\[
\\text{Recall} = \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} + \\text{False Negatives (FN)}}
\\]

**Purpose:**  
Recall measures how many actual disease cases the model correctly identifies. It's **critical when missing a positive case (false negative)** can have serious consequences, such as delayed treatment.

**Example Use Case:**  
In cancer detection, we want to **maximize recall** to avoid missing any real cases, even if it means having more false alarms.

---

## 🚨 2. False Positive Rate (FPR)

**Formula:**

\\[
\\text{FPR} = \\frac{\\text{False Positives (FP)}}{\\text{False Positives (FP)} + \\text{True Negatives (TN)}}
\\]

**Purpose:**  
FPR tells us how many healthy individuals were wrongly classified as sick. **Lower FPR is preferred**, especially when false alarms cause emotional stress, unnecessary tests, or costly follow-ups.

---

## 📈 3. ROC Curve (Receiver Operating Characteristic)

The **ROC curve** is a graphical representation that shows the **trade-off between Recall (TPR)** and **False Positive Rate (FPR)** at different classification thresholds.

### How it Works:
- **Y-axis**: True Positive Rate (Recall)
- **X-axis**: False Positive Rate
- Each point on the ROC curve corresponds to a **different threshold** used to decide whether a predicted probability counts as a positive prediction.

### Why It’s Useful:
- Helps visualize how the model performs across various thresholds.
- Supports choosing a threshold that balances high recall and low FPR.
- Includes a single summary score: **AUC (Area Under the Curve)**
  - AUC = 1.0 → perfect model
  - AUC = 0.5 → random guessing