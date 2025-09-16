import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os

print("=== Regression Tool ===")
print("You can either:\n1. Enter data points manually\n2. Load from a CSV file in the same folder.\n")

X, y = [], []

choice = input("Type 'manual' for manual entry or 'file' to load CSV: ").strip().lower()

if choice == "file":
    files = [f for f in os.listdir() if f.endswith(".csv")]
    if not files:
        print("No CSV files found in this folder.")
        exit()
    
    print("\nAvailable CSV files:")
    for i, f in enumerate(files, start=1):
        print(f"{i}. {f}")
    
    idx = int(input("Select file number: ")) - 1
    df = pd.read_csv(files[idx])

    print("\nColumns found:", list(df.columns))
    x_col = input("Enter column name for X: ").strip()
    y_col = input("Enter column name for Y: ").strip()

    try:
        X = df[[x_col]].values
        y = df[y_col].values
    except:
        print("Invalid column names.")
        exit()

else:
    print("Enter your data points (x and y values). Type 'done' when finished.\n")
    while True:
        entry = input("Enter x,y (or 'done' to finish): ").strip()
        if entry.lower() == "done":
            break
        try:
            x_val, y_val = entry.split(",")
            X.append([float(x_val)])
            y.append(float(y_val))
        except:
            print(" Please enter values in the format: x,y (e.g. 2,4)")

    if len(X) < 2:
        print("ERROR: Need at least 2 points for regression.")
        exit()

# Train with sklearn (for prediction loop)
model = LinearRegression()
model.fit(X, y)

# Train with statsmodels (for diagnostics)
X_sm = sm.add_constant(X)  # add intercept
ols_model = sm.OLS(y, X_sm).fit()

print("\n=== Model Summary ===")
print(f"Equation: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
print(f"R²: {ols_model.rsquared:.4f}  (Explained variance)")
print(f"Coefficient (Beta): {ols_model.params[1]:.4f}")
print(f"Intercept: {ols_model.params[0]:.4f}")
print(f"Standard Error: {ols_model.bse[1]:.4f}")
print(f"T-statistic: {ols_model.tvalues[1]:.4f}")
print(f"P-value: {ols_model.pvalues[1]:.4f}")

print("\n--- Threshold summary ---")
print("R²: Context dependent, not a primary filter")
print("P-value: <0.05 acceptable, <0.01 robust")
print("T-statistic: >2 baseline, >3 strong")
print("Standard error: Must allow |t| ≥ 2")
print("Coefficient: Must be economically interpretable\n")

# Prediction loop
while True:
    new_x = input("\nEnter a new x value to predict y (or 'exit' to quit): ").strip()
    if new_x.lower() == "exit":
        break
    try:
        prediction = model.predict([[float(new_x)]])[0]
        print(f" Predicted y: {prediction:.4f}")
    except:
        print("Please enter a number.")
