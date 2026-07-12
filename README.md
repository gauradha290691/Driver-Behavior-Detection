🚗 Driver Behavior Detection using Machine Learning
A machine learning project that classifies driving behavior as Harsh or Normal using vehicle telemetry data. The model is built using Logistic Regression and demonstrates a complete machine learning workflow---from data preprocessing to model deployment.
See it in action https://drive.google.com/file/d/1gajFx4uXEvinqWHKJB9o7lWfzAltebkK/view

📌 Project Overview
Driver behavior analysis plays a vital role in:
    • Fleet Management
    • Usage-Based Insurance (UBI)
    • Driver Safety Monitoring
    • Connected Vehicles
    • Automotive Telematics
    • Smart Mobility Solutions
This project uses sensor-based driving data to identify harsh driving events that may indicate aggressive acceleration, braking, or unsafe vehicle handling.

🎯 Objectives
    • Preprocess raw driving behavior data
    • Train a machine learning classification model
    • Evaluate model performance
    • Save the trained model for future predictions
    • Demonstrate an end-to-end machine learning pipeline

🛠 Technologies Used
Technology     Purpose

Python         Programming Language 
Pandas         Data Processing 
NumPy          Numerical Computing 
Scikit-learn   Machine Learning 
Joblib         Model Serialization


📂 Project Structure
Driver-Behavior-Detection/
│
├── DriverBehaviourData.py
├── test.py
├── DriverBehaviour.pkl
├── Scalar.pkl
├── README.md
└── .gitignore

⚙️ Installation
git clone  https://github.com/gauradha290691/Driver-Behavior-Detection.git
cd Driver-Behavior-Detection
pip install pandas numpy scikit-learn joblib

▶️ Running the Project
Train the model:
python DriverBehaviourData.py
Test the model:
python test.py

🔄 Machine Learning Workflow
    1. Load dataset
    2. Remove missing values
    3. Split features and labels
    4. Train/Test split
    5. Standardize features
    6. Train Logistic Regression model
    7. Evaluate performance
    8. Save model and scaler

📊 Model Evaluation
    • Training Accuracy
    • Testing Accuracy
    • Classification Report
    • Confusion Matrix

💾 Model Files
    • DriverBehaviour.pkl -- Trained model
    • Scalar.pkl -- Saved StandardScaler

📈 Future Improvements
    • Random Forest
    • XGBoost
    • LightGBM
    • TensorFlow/Keras
    • Hyperparameter Tuning
    • Edge AI Deployment
    • Apple Core ML Deployment

⚠️ Dataset Information
The dataset may not be included in this repository due to licensing restrictions. Download it from the original source and place it in the project directory as:
driving_behavior_training_data.csv
https://github.com/jair-jr/driverBehaviorDataset/tree/master

👨‍💻 Author
Gaurav Jadhao
Product Owner | AI & Machine Learning Enthusiast | Automotive Platform Engineering
GitHub: https://github.com/gauradha290691
LinkedIn:https://www.linkedin.com/in/gaurav-jadhao-b7258a52/

⭐ Support
If you found this project useful, consider giving it a ⭐ on GitHub.
