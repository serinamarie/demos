import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import Any

def create_data():
    df = pd.read_csv("titanic.csv")
    df = df.drop(["Name"], axis = 1)
    df["Sex"] = pd.factorize(df["Sex"])[0]
    y = df["Survived"]
    X = df.drop("Survived", axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    fill_age = X_train["Age"].mean()
    X_train["Age"] = X_train["Age"].fillna(fill_age)
    X_test["Age"] = X_test["Age"].fillna(fill_age)
    return X_train, X_test, y_train, y_test

def get_models():
    return [LogisticRegression(random_state=42),
            KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), 
            RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
            RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)]

def train_model(model: Any, X_train, X_test, y_train, y_test):
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"model": model.__class__.__name__, "params": model.get_params(), "accuracy": acc}

def get_results(results):
    res = pd.DataFrame(results)
    return res

def distributed_flow(): # change name of function
    X_train, X_test, y_train, y_test = create_data()
    models = get_models()
    training_runs = [train_model(model, X_train, X_test, y_train, y_test) for model in models]
    return get_results(training_runs)


if __name__ == "__main__":
   
    print(distributed_flow())