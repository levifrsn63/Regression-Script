import numpy as np
from sklearn.linear_model import LinearRegression

print("=== Regression Tool ===")
print("Enter your data points (x and y values). Type 'done' when finished.\n")

X, y = [], []

# Collect data points
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

# Train model
model = LinearRegression()
model.fit(X, y)

print("\n Model trained!")
print(f"Equation: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")

# Predict loop
while True:
    new_x = input("\nEnter a new x value to predict y (or 'exit' to quit): ").strip()
    if new_x.lower() == "exit":
        break
    try:
        prediction = model.predict([[float(new_x)]])[0]
        print(f" Predicted y: {prediction:.2f}")
    except:
        print("Please enter a number.")
