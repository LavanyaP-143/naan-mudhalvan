import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_xgb(df: pd.DataFrame, target: str = "pm25"):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"XGBoost RMSE: {rmse:.2f}")
    joblib.dump(model, "xgb_model.joblib")
    return model

if __name__ == "__main__":
    df = load_data("data/air_quality.csv")
    train_xgb(df)
# naan-mudhalvan
