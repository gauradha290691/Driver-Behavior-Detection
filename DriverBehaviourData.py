import joblib
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix




DriverBehDF=pd.read_csv("driving_behavior_training_data.csv")
print(DriverBehDF)

DriverBehCleaned=DriverBehDF.dropna()
print(DriverBehCleaned)

x=DriverBehCleaned.drop(columns=["label","trip_id","window_start_sec"])
y=DriverBehCleaned["label"]
print(x)
print(y)


train_x,test_x,train_y, test_y=train_test_split(x,y,train_size=0.8,random_state=23)
print(test_y)

Scalar=StandardScaler()
train_x_scaled=Scalar.fit_transform(train_x)
test_x_scaled=Scalar.fit_transform(test_x)
print(train_x_scaled)

model=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=23)
model.fit(train_x_scaled,train_y)
pred_y=model.predict(train_x_scaled)

accuracy=accuracy_score(pred_y,train_y)
print(accuracy)

pred_y=model.predict(test_x_scaled)

accuracy=accuracy_score(pred_y,test_y)
print("test accuracy",accuracy)

joblib.dump(Scalar,"Scalar.pkl")
joblib.dump(model, "DriverBehaviour.pkl")

print(classification_report(test_y, pred_y))
print(confusion_matrix(test_y, pred_y, labels=["Harsh", "normal"]))






