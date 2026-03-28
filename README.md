### README.md

# STL-10 Constant Predictor Model

## 🚀 Overview

This project implements a **minimal deep learning model** for the STL-10 dataset that always predicts a fixed class (class 0), regardless of input.

The goal of this implementation is to **minimize model size** to an extreme level for experimentation with size-based evaluation metrics.

---

## 🧠 Model Architecture

The model is intentionally simple:

- No learnable parameters
- No convolutional layers
- No training required

### Behavior:

- For any input image, the model outputs logits where:
  - Class `0` has the highest score
  - All other classes are zero

This ensures the model **always predicts class 0**.

---

## 📦 Model Size

- Extremely small (~1–2 KB)
- No weights or parameters stored
- Designed for **maximum compression efficiency**

---

## 📊 Expected Performance

Since STL-10 has 10 balanced classes:

- Expected Accuracy: **~10%**
- Correct predictions only when true label = class 0

---

## 🏋️ Training Procedure

No training is required.

The model is deterministic and does not learn from data.

---

## ▶️ How to Run

### 1. Install dependencies

```

pip install -r requirements.txt

```

### 2. Run evaluation

```

python test.py --data_path ./data

```

---

## 📂 Project Structure

```

project_root/
│── model.py
|── train.py
│── test.py
│── README.md
│── requirements.txt

```

---

## ⚠️ Limitations

- Extremely low accuracy (~10%)
- Does not learn from data
- Not suitable for real-world applications
- Intended only for **baseline and experimental purposes**

---

## 🎯 Use Case

This model is useful for:

- Understanding evaluation metrics
- Testing scoring systems
- Exploring extreme model compression limits

---
