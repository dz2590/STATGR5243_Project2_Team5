````markdown
# STATGR5243_Project2_Team5  
STATGR5243 Team 5 — Project 2 Code Repository  

# 📊 End-to-End Data Analysis Tool

An interactive web application built with **Shiny for Python** that enables users to complete a full data analysis workflow — including **data loading, preprocessing, feature engineering, and exploratory data analysis (EDA)** — within a unified interface.

---

# 🚀 Quick Start

### 1. Install dependencies
```bash
pip install shiny pandas numpy matplotlib scikit-learn openpyxl pyarrow
````

### 2. Run the application

```bash
python -m shiny run --reload app.py
```

### 3. Open in browser

```
http://127.0.0.1:8000
```

---

# 🧭 Step-by-Step User Guide

This section provides a practical, end-to-end workflow for using the tool.

---

## 🔹 Step 1 — Load Data

### Option A: Built-in dataset (recommended for quick testing)

1. Go to **Load Data**
2. Select **Built-in dataset**
3. Choose:

   * Diabetes
   * Breast Cancer

You will immediately see:

* Dataset summary (rows, columns, missing values)
* Data preview (first 10 rows)

---

### Option B: Upload your own dataset

1. Select **Upload file**
2. Upload supported formats:

   * `.csv`, `.xlsx`, `.parquet`, `.json`

⚠️ Notes:

* Ensure correct file format
* Large files may take time to load

---

## 🔹 Step 2 — Clean & Preprocess

This stage prepares your dataset for analysis.

### Recommended workflow:

### 1. Remove duplicates

* Enable: **Remove duplicate rows**

---

### 2. Drop unnecessary columns

* Use: **Select columns to remove**
* Examples:

  * ID columns
  * Irrelevant variables

---

### 3. Handle missing values

| Method    | Use case             |
| --------- | -------------------- |
| Drop rows | Few missing values   |
| Mean      | Continuous variables |
| Median    | Skewed data          |
| Mode      | Categorical data     |

---

### 4. Handle outliers

| Method       | Description                 |
| ------------ | --------------------------- |
| IQR clipping | Limits extreme values       |
| Remove rows  | Removes outliers completely |

---

### 5. Scale features

| Method             | Use case                      |
| ------------------ | ----------------------------- |
| Standard (z-score) | Machine learning              |
| Min-Max            | Visualization / normalization |

---

### 6. Encode categorical variables

* Enable: **One-hot encoding**

---

### ✔ Check results

Review:

* Preprocessing summary
* Before vs After comparison
* Processed data preview

---

### ✔ Export processed dataset

Click:

```
Download Processed CSV
```

---

## 🔹 Step 3 — Feature Engineering

Enhance your dataset by creating new variables.

---

### ① Single Column Transformation

Example:

* Column: `age`
* Method: `log`

Result:

```
log_age
```

Use for:

* Reducing skewness
* Normalization

---

### ② Two-Column Interaction

Example:

* Column A: `income`
* Column B: `expenses`
* Operation: `ratio_pct`

Result:

```
income_pct
```

Use for:

* Ratios
* Feature relationships

---

### ③ Binning (Discretization)

Example:

* Column: `age`
* Bins: `5`

Result:

```
age_bin5
```

Use for:

* Grouping continuous variables
* Modeling

---

### ✔ Key features

* Automatic column naming (no overwriting)
* Error handling for invalid operations
* Before vs after distribution visualization

---

### ✔ Reset all features

Click:

```
Reset All Features
```

---

### ✔ Export final dataset

Click:

```
Download Final CSV
```

---

## 🔹 Step 4 — Exploratory Data Analysis (EDA)

Analyze and visualize your data.

---

### 📊 Available plots

| Plot                | Purpose               |
| ------------------- | --------------------- |
| Histogram           | Distribution          |
| Boxplot             | Outlier detection     |
| Scatterplot         | Relationships         |
| Bar chart           | Categorical frequency |
| Correlation heatmap | Feature relationships |

---

### 📌 Example workflow

1. Select **Scatterplot**
2. Choose:

   * X variable: `age`
   * Y variable: `income`
3. Analyze relationship between variables

---

### 🔍 Filtering

1. Select a numeric column
2. Adjust the range slider
3. Dataset updates dynamically

---

### 📊 Additional outputs

* Summary statistics
* Correlation matrix
* Filtered dataset preview

---

# 🧩 Project Structure

```
.
├── app.py              # Main Shiny application
├── README.md          # Documentation
└── data/              # Optional datasets
```

---

# ⚙️ Technologies Used

* Python
* Shiny for Python
* Pandas / NumPy
* Matplotlib
* Scikit-learn

---

# 💡 Design Highlights

* Reactive programming (automatic updates)
* Modular data processing pipeline
* Clean multi-tab UI design
* Built-in error handling
* Interactive and user-friendly workflow

---

# ⚠️ Limitations

* No advanced machine learning models included
* Performance may decrease with large datasets
* Workflow steps are not saved persistently

---

# 🚀 Future Improvements

* Add machine learning modules (classification/regression)
* Support saving/loading workflows
* Use interactive visualization libraries (e.g., Plotly)
* Deploy application to cloud platforms

---

# 👨‍💻 Authors

STATGR5243 Team 5

---

# 📜 License

This project is for academic use only.

```
```
