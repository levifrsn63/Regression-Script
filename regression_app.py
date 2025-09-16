import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Simple Regression Tool")

st.write("Enter data points (x and y) and fit a regression line.")

# Data input
data = st.text_area("Enter your data (format: x,y per line)", "1,2\n2,4\n3,5\n4,4\n5,5")

# Parse data
X, y = [], []
for line in data.strip().split("\n"):
    try:
        x_val, y_val = line.split(",")
        X.append([float(x_val)])
        y.append(float(y_val))
    except:
        pass

if len(X) >= 2:
    model = LinearRegression()
    model.fit(X, y)

    st.success(f"✅ Model trained: y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")

    # Prediction
    new_x = st.number_input("Enter a new x value to predict y", value=6.0)
    pred = model.predict([[new_x]])[0]
    st.write(f"📌 Predicted y: **{pred:.2f}**")

    # Plot
    X_np = np.array(X)
    y_np = np.array(y)
    y_line = model.predict(X_np)

    fig, ax = plt.subplots()
    ax.scatter(X_np, y_np, color="blue", label="Data points")
    ax.plot(X_np, y_line, color="red", label="Regression line")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ Please enter at least 2 data points.")
