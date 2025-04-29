# task_3
Predicting house prices using Multiple Linear Regression on the real Housing dataset using Python and scikit-learn
#  House Price Prediction using Linear Regression (Real Dataset)

This project applies **Multiple Linear Regression** to predict house prices based on various features like **area**, **number of bedrooms**, **bathrooms**, **furnishing status**, and other facilities using the real **Housing.csv** dataset.

It is built as part of **Task 3** for an **AI & ML Internship**.

---

## Dataset Description

- **Dataset Source**: [Kaggle Housing Price Prediction Dataset](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- **File Used**: `Housing.csv`
- **Features**:
  - `area`: Area of the house in square feet
  - `bedrooms`: Number of bedrooms
  - `bathrooms`: Number of bathrooms
  - `stories`: Number of stories
  - `mainroad`: Access to the main road (yes/no)
  - `guestroom`: Availability of guest room (yes/no)
  - `basement`: Availability of basement (yes/no)
  - `hotwaterheating`: Hot water heating availability (yes/no)
  - `airconditioning`: Air conditioning availability (yes/no)
  - `parking`: Number of parking spaces
  - `prefarea`: Preferred area (yes/no)
  - `furnishingstatus`: Furnishing status (furnished, semi-furnished, unfurnished)
  - `price`: Target variable — Price of the house

---

## Project Goals

- Perform **data preprocessing** (handling categorical features)
- Apply **Multiple Linear Regression**
- **Split** dataset into **training** and **testing** sets
- **Train** the model on the training data
- **Predict** house prices on the test data
- **Evaluate** model performance:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score**
- **Visualize**:
  - Actual vs Predicted house prices
- **Interpret** coefficients to understand the impact of each feature

---

##  Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Workflow

### Step 1: Import Libraries
- Imported Pandas, Matplotlib, Seaborn, and scikit-learn packages.

### Step 2: Load Dataset
- Loaded `Housing.csv` into a Pandas DataFrame.
- Checked data types and null values.

### Step 3: Data Preprocessing
- Mapped categorical **yes/no** columns to **1/0**.
- Applied **One-Hot Encoding** to the `furnishingstatus` column using `get_dummies()`.
- Verified dataset is fully numeric and ready for modeling.

### Step 4: Define Features and Target
- Selected independent variables (`X`) and dependent variable (`y = price`).

### Step 5: Split the Data
- Used **train_test_split()** to split data:
  - 80% for training
  - 20% for testing

### Step 6: Train the Model
- Trained **Linear Regression** model using scikit-learn.

### Step 7: Make Predictions
- Made predictions on the test set.

### Step 8: Evaluate the Model
- Evaluated model using:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score (Accuracy Measure)**

### Step 9: Visualize Predictions
- Plotted **Actual vs Predicted** prices scatter plot.
- Ideal predictions align along the **45-degree diagonal**.

### Step 10: Interpret Coefficients
- Explained how each feature affects house price prediction.
- Higher positive coefficients mean a greater positive impact.

---

## Model Evaluation Metrics

| Metric | Meaning | Why Important |
|:------|:--------|:--------------|
| **MAE** | Average of absolute errors | How much, on average, our predictions are off |
| **MSE** | Average of squared errors | Punishes larger errors more heavily |
| **R² Score** | Goodness of fit | 1.0 means perfect prediction, closer to 0 is bad |

---

## Files Included

| File | Description |
|:-----|:------------|
| `HousePrice_Real_LinearRegression.ipynb` | Full Jupyter Notebook with code and outputs |
| `Housing.csv` | Housing dataset used for training and testing |
| `README.md` | Project explanation |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-linear-regression-real.git
   cd house-price-linear-regression-real
