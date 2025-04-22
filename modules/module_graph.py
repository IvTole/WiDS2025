import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
import seaborn as sns

# MLFlow
import mlflow
from mlflow.models.signature import infer_signature

# External libraries
from module_path import plots_data_path 

#Sets the path to save the plots
plots_path = plots_data_path()

def graph_tree (model):
    # Plots a tree from the random forest
    plt.figure(figsize=(20,10))
    plot_tree(model.estimators_[0], filled=True) 
    plt.title("Individual Tree from the Random Forest")
    plt.savefig(os.path.join(plots_path,'decision_tree.png'))
    #mlflow.log_artifact(os.path.join(plots_path,'decision_tree.png'))
    #plt.show()

def graph_confusion_matrix(cm, model, tag, labels=None):
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels if labels is not None else "auto",
                yticklabels=labels if labels is not None else "auto")
    plt.title(f"Confusion Matrix - {model} {tag}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    
    filename = f"conf_matrix_{model}_{tag}.png"
    full_path = os.path.join(plots_path, filename)
    plt.savefig(full_path)
    plt.close()

    mlflow.log_artifact(full_path)