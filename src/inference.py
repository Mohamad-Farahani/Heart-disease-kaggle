
import pandas as pd
import joblib

def predict():

    model = joblib.load("models/model.pkl")

    test = pd.read_csv("data/test.csv")

    preds = model.predict(test)

    pd.DataFrame(preds,columns=["prediction"]).to_csv("submission.csv",index=False)

if __name__=="__main__":
    predict()
