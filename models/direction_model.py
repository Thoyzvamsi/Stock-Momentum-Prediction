from features.engineering import Features_Engineering
import pandas as pd
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

class Direction_Model:
    def __init__(self,data):
        self.data = Features_Engineering(data).features()

    def training_model(self):
        split = int(len(self.data) * 0.8)

        X = self.data[
            ["returns",
            "Volume_imbalance",
            "volume_surge",	
            "candle_size",
            "tick_momentum",
            "vwap_distance",
            "body_wick_ratio",
            "volatility_percentile",
            "consecutive_same_direction",
            "price_structure"
            ]
        ]

        y = self.data["Target"]


        X_train = X[:split]
        X_test = X[split:]

        y_train = y[:split]
        y_test = y[split:]

        model = LGBMClassifier(
            class_weight="balanced",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        df = pd.DataFrame(X_test)
        df["date"] = self.data["date"]
        df["time"] = self.data["time"]
        df["pred"] = preds
        df["prob"] = probs[:,2]
        df["close"] = self.data["close"]

        return df
