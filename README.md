
# STATGR5243 Project 2 Team 5  

# End-to-End Data Analysis Tool

An interactive web application built with **Shiny in Python** that allows users to complete a data analysis workflow that includes data loading, preprocessing, feature engineering, and exploratory data analysis (EDA). 

---
# Project Structure

```
.
├── app.py              # Main Shiny application
└── README.md          # Documentation
```
---
# Quick Start

## # Requirements
Python Version: 3.10-3.12
Packages: shiny, pandas, numpy, matplotlib, seaborn, scikit-learn, pyreadr, openpyxl, pyarrow

## Set up and Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Create a virtual environment to prevent package conflicts
```bash
python -m venv .venv
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Windows:
```bash
.venv\Scripts\activate
```

### 2. In your terminal, install dependencies
```bash
pip install shiny pandas numpy matplotlib seaborn scikit-learn pyreadr openpyxl pyarrow
```

### 3. Run the application

```bash
shiny run --reload project2_app.py
```

### 3. Open in browser the local URL shown in the terminal

```
http://127.0.0.1:8000
```
---
## Supported Input Formats
The app can load CSV, Excel, Parquet, and RDS files.
For uploaded files, the app automatically detects the file extension and reads the dataset accordingly.

# How to Use: 

## Step 1 — Load Data

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

Notes:
* Ensure correct file format
* Large files may take time to load

---

## Step 2 — Exploratory Data Analysis (EDA)
Use the EDA tab to inspect the data dictionary, filter rows by selected column vales, and view descriptive statistics. 

## Step 3 - Visualizations
Choose a plot type, select variables, and inspect the numeric correlations between two variables.
The available plot types are histogram, box plot, scatter plot, bar chart, and correlation matrix. 

## Step 4 - Data Cleaning and Preprocessing
Remove duplicates or outliers, impute missing values, encode categorical variables, and scale numeric features. Preview the processed dataset and then download the cleaned data as a csv file.

## Step 5 - Feature Engineering
Transform one numeric column at a time, combine two numeric columns, or bin / discretize numeric variables. Review the changes by inspecting the before and after distributions of these transformations, and download the final engineered dataset. 


--
---

```
