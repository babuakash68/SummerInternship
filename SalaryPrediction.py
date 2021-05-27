import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
data = pd.read_csv("/root/salary.csv")

x = data["YearsExperience"]
y = data["Salary"]
print(x)
print(y)
model = LinearRegression()
x =  x.values
x = x.reshape(-1,1)
print("Predicted Salary = ")
model.fit(x,y)
print(model.predict([[10]]))
import joblib
joblib.dump(model,"result.pkl")

