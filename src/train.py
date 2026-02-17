from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
def train(xscaled, ytrain):
    model = RandomForestClassifier()
    model.fit(xscaled, ytrain)
    return model
def save_model(model, path):
    joblib.dump(model, path)