import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

Model1=joblib.load("DriverBehaviour.pkl")
Scalar=joblib.load("Scalar.pkl")

df=pd.read_csv("driving_behavior_training_data.csv")

x=df.drop(columns=["label","trip_id","window_start_sec"])
y=df["label"]

x_scaled=Scalar.transform(x)

pred_y=Model1.predict(x_scaled)

print(accuracy_score(pred_y,y))


print(np.count_nonzero(pred_y=="normal"))