# Mobile Data Usage Predictor

A machine learning project that predicts mobile data usage using:

* **Simple Linear Regression** (baseline model)
* **Multiple Linear Regression** (improved model)

---

## 📂 Project Structure

```
Mobile-Data-Usage-Predictor/
│
├── simple_linear_regression/
│   └── mobile_data.py
│
├── multiple_linear_regression/
│   └── data_usage.py
│
├── data.csv
├── README.md
```

---

## 🧠 Models Overview

### 1. Simple Linear Regression

* Uses **one feature**:

  * Screen Time (hours/day)
* Predicts:

  * Data Usage (MB/day)

**Purpose:**
Understand the basic linear relationship between screen time and data usage.

---

### 2. Multiple Linear Regression

* Uses multiple features:

  * Screen Time
  * App Usage
  * Battery Drain
  * Apps Installed

**Purpose:**
Improve prediction accuracy by considering multiple factors.

---

## ⚙️ Workflow

1. Load dataset using pandas
2. Clean and preprocess data
3. Select relevant features
4. Split into training and testing sets
5. Train Linear Regression model
6. Evaluate using metrics
7. Make predictions
8. Visualize results

---

## 📊 Model Performance

| Model                      | R² Score   | MAE (MB/day) |
| -------------------------- | ---------- | ------------ |
| Simple Linear Regression   | ~0.85–0.90 | ** 158.37** |
| Multiple Linear Regression | **0.906**  | **137**      |

**Interpretation:**

* Higher R² → better model fit
* Lower MAE → more accurate predictions

---

## 📌 Key Insights

* Data usage increases with screen time
* Multiple features significantly improve accuracy
* Real-world behavior is influenced by multiple variables, not just one


## 👤 Author

**Shriya Gudkaa**
