#!/usr/bin/env python
# coding: utf-8

# In[5]:


from src.preprocess import load_model, scaler
from src.train import train, save_model
from src.evaluate import eval_model, GridSearch

#GridSearchCV gives the same Accuracy score as the base model in this classification project

def main():
    xtrain, xtest, ytrain, ytest= load_model("datasets/heart-disease.csv")
    x_train, x_test, scaler_instance = scaler(xtrain, xtest)
    with open("data.json", "w") as f:
        json.dump({"xtrain":x_train, "xtest":x_test, "ytrain":ytrain, "ytest":ytest}, f)
        print("Saved data")
    model = train(x_train, ytrain)
    preds, accuracy, f1, recall, report = eval_model(model, x_test, ytest)
    gs_model, best_params, grid_score = GridSearch(x_train, x_test, ytrain, ytest)
    
    print("x------------------Base Model Results-------------------x")
    print(f"Overall accuracy of the Model : {accuracy}")
    print(f"F1 score : {f1}")
    print(f"Recall score : {recall}\n")
    
    print("x------------------Grid Search Results-------------------x")
    print(f"Best parameters : {best_params}")
    print(f"GridSearchCV Score : {grid_score}\n")
    
    print("x------------------Classification Report-------------------x\n")
    print(report)
    print("x----------------------------------------------------------x")

    #For experimentation
    return gs_model, x_test, ytest, preds

if __name__ == "__main__":
    main()


# In[ ]:




