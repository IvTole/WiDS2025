#Imported libraries.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Parameters of each model.
model_params = {
    "lr": {
        "solver": "lbfgs",
        "max_iter": 5000
    },

    "rf": {
        "n_estimators": 1000,
        "criterion": "gini",
        "max_depth": 10,
        "random_state": 42,
        "bootstrap": True
    }
}

#Functions to call each model.
def get_lr():
    return LogisticRegression(**model_params['lr'])

def get_rf():
    return RandomForestClassifier(**model_params['rf'])