import joblib
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------------
# Load the driving behavior dataset
# --------------------------------------------------------
DriverBehDF = pd.read_csv("driving_behavior_training_data.csv")
print(DriverBehDF)

# --------------------------------------------------------
# Remove rows containing missing values
# --------------------------------------------------------
DriverBehCleaned = DriverBehDF.dropna()
print(DriverBehCleaned)

# --------------------------------------------------------
# Separate input features (X) and target label (Y)
# Exclude columns that are not useful for prediction
# --------------------------------------------------------
x = DriverBehCleaned.drop(columns=["label", "trip_id", "window_start_sec"])
y = DriverBehCleaned["label"]

print(x)
print(y)

# --------------------------------------------------------
# Split the dataset into training and testing sets
# 80% for training and 20% for testing
# --------------------------------------------------------
train_x, test_x, train_y, test_y = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=23
)

print(test_y)

# --------------------------------------------------------
# Standardize numerical features
# --------------------------------------------------------
Scalar = StandardScaler()

# Fit the scaler using training data
train_x_scaled = Scalar.fit_transform(train_x)

# NOTE:
# Ideally use Scalar.transform(test_x)
# Keeping the original code unchanged here
test_x_scaled = Scalar.fit_transform(test_x)

print(train_x_scaled)

# --------------------------------------------------------
# Create the Logistic Regression classifier
# class_weight='balanced' helps when classes are imbalanced
# --------------------------------------------------------
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=23
)

# --------------------------------------------------------
# Train the model
# --------------------------------------------------------
model.fit(train_x_scaled, train_y)

# --------------------------------------------------------
# Predict on training data
# --------------------------------------------------------
pred_y = model.predict(train_x_scaled)

# Calculate training accuracy
accuracy = accuracy_score(pred_y, train_y)
print("Training Accuracy:", accuracy)

# --------------------------------------------------------
# Predict on testing data
# --------------------------------------------------------
pred_y = model.predict(test_x_scaled)

# Calculate testing accuracy
accuracy = accuracy_score(pred_y, test_y)
print("Testing Accuracy:", accuracy)

# --------------------------------------------------------
# Save the trained model and scaler
# --------------------------------------------------------
joblib.dump(Scalar, "Scalar.pkl")
joblib.dump(model, "DriverBehaviour.pkl")

# --------------------------------------------------------
# Print detailed evaluation metrics
# --------------------------------------------------------
print(classification_report(test_y, pred_y))

# Display the confusion matrix
print(confusion_matrix(test_y, pred_y, labels=["Harsh", "normal"]))
