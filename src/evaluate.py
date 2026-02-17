from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
def eval_model(model, xtest, ytest):
    preds = model.predict(xtest)
    accuracy = accuracy_score(ytest, preds)
    f1 = f1_score(ytest, preds)
    recall = recall_score(ytest, preds)
    classification_rpt = classification_report(ytest, preds)
    return accuracy, f1, recall, classification_rpt