import pandas as pd
import xgboost as xgb
from features import get_features

df = pd.read_csv("data/thumbnails.csv")
print("training on", len(df), "thumbnails")

X = [get_features(p) for p in df["file"]]
y = df["engagement_rate"]

model = xgb.XGBRegressor(n_estimators=200, max_depth=4)
model.fit(X, y)
model.save_model("thumb_model.json")
print("saved thumb_model.json")
