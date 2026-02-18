from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_model(path):
    df = pd.read_csv(path)
    x = df.drop("target", axis=1)
    y = df["target"]
    model = RandomForestClassifier()
    crossval = cross_val_score(model, x, y, cv=5)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=32)
    return xtrain, xtest, ytrain, ytest, crossval.mean()

def scaler(xtrain, xtest):
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled, scaler