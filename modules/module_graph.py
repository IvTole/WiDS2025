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

def graph_tree (model, tag):
    # Plots a tree from the random forest
    plt.figure(figsize=(20,10))
    plot_tree(model.estimators_[0], filled=True) 
    plt.title(f"Random_Forest_{tag}")
    plt.savefig(os.path.join(plots_path,f"Random_forest_{tag}.png"))
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

def graph_roc_curve(fpr, tpr, auc, model_name, tag):
    """
    Genera y guarda la gráfica de la Curva ROC.

    :param fpr: Tasa de Falsos Positivos
    :param tpr: Tasa de Verdaderos Positivos
    :param auc: Área bajo la Curva ROC
    :param model_name: Nombre del modelo (para el título y el nombre del archivo)
    :param tag: Tag o etiqueta (para el nombre del archivo)
    """
    plt.figure()
    plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {model_name} - {tag}') # Incorporar nombre del modelo y tag
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_data_path(), f'roc_curve_{model_name}_{tag}.png')) # Incorporar nombre del modelo y tag en el nombre del archivo
    plt.close()