import pandas as pd
import xgboost as xgb
from captions import caption_features

df = pd.read_csv("data/captions.csv")
print("training on", len(df), "captions")

X = [caption_features(c) for c in df["caption"]]
y = df["engagement_rate"]

model = xgb.XGBRegressor(n_estimators=150, max_depth=3)
model.fit(X, y)
model.save_model("caption_model.json")
print("saved caption_model.json")
