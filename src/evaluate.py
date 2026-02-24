from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def eval_model(model, xtest, ytest):
    preds = model.predict(xtest)
    accuracy = accuracy_score(ytest, preds)
    f1 = f1_score(ytest, preds)
    recall = recall_score(ytest, preds)
    classification_rpt = classification_report(ytest, preds)
    return accuracy, f1, recall, classification_rpt


#Using GridSearchCV to get optimum score by tuning the hyperparameters
def GridSearch(xtrain, xtest, ytrain, ytest):
    gs_grid = {"C":[0.001, 0.01, 0.1, 1, 10, 100],
              "penalty":["l2"],
              "solver":['lbfgs','saga'],
              "class_weight": [None, 'balanced']}

    gs_log_reg = GridSearchCV(LogisticRegression(), param_grid = gs_grid, cv = 5, verbose = True)
    gs_log_reg.fit(xtrain, ytrain)
    return gs_log_reg.best_params_, gs_log_reg.score(xtest, ytest)