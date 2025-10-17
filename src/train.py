import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yaml, os, json

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/processed/zameen-updated.csv")
X = df.drop('price', axis=1, errors='ignore') 
y = df['price'] if 'price' in df.columns else df.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['split']['test_size'],
    random_state=params['train']['random_state']
)


model = RandomForestRegressor(
    n_estimators=params['train']['n_estimators'],
    random_state=params['train']['random_state']
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"MSE: {mse}")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
os.makedirs("metrics", exist_ok=True)
json.dump({"mse": mse}, open("metrics/metrics.json", "w"))
