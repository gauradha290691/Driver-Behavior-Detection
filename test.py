import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

# --------------------------------------------------------
# Load the saved machine learning model and scaler
# --------------------------------------------------------
Model1 = joblib.load("DriverBehaviour.pkl")
Scalar = joblib.load("Scalar.pkl")

# --------------------------------------------------------
# Load the driving behavior dataset
# --------------------------------------------------------
df = pd.read_csv("driving_behavior_training_data.csv")

# --------------------------------------------------------
# Separate input features and target labels
# Remove columns that are not used for prediction
# --------------------------------------------------------
x = df.drop(columns=["label", "trip_id", "window_start_sec"])
y = df["label"]

# --------------------------------------------------------
# Standardize the input features using the
# scaler generated during model training
# --------------------------------------------------------
x_scaled = Scalar.transform(x)

# --------------------------------------------------------
# Predict driving behavior using the trained model
# --------------------------------------------------------
pred_y = Model1.predict(x_scaled)

# --------------------------------------------------------
# Calculate and display prediction accuracy
# --------------------------------------------------------
print("Model Accuracy:", accuracy_score(pred_y, y))

# --------------------------------------------------------
# Count the total number of samples predicted
# as 'normal' driving behavior
# --------------------------------------------------------
print("Number of Normal Driving Predictions:",
      np.count_nonzero(pred_y == "normal"))
