
import pandas as pd
import lightgbm as lgb
import joblib

def load_data():
    return pd.read_csv("data/train.csv")

def train():

    df = load_data()

    X = df.drop("target",axis=1)
    y = df["target"]

    model = lgb.LGBMClassifier()

    model.fit(X,y)

    joblib.dump(model,"models/model.pkl")

if __name__=="__main__":
    train()
