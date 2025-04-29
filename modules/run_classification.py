# Standar libraries
from datetime import datetime

# Scikit learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA



# External modules
from module_path import test_data_path, train_data_path, plots_data_path
from module_data import Dataset
from module_graph import graph_tree
from module_model import ModelEvaluation, ModelEvaluationXG, ModelSubmission



def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    # create a Dataset object
    df = Dataset()

    # train dataframe, test dataframe, y targets dataframe
    df_train, df_test, labels = df.load_data_frame_imputed()

    #Elimina columnas que no tienen una buena correlación con el resultado
    df_train_select, df_test_select, labels = df.load_relevant_data()

    # Genera train dataframe, test dataframe, versión estandarizada para utilizar en futuros modelos 
    df_train_std, df_test_std = df.load_data_frame_standardized()

    # define array of target variables for the model
    targets = ['ADHD_Outcome',  'Sex_F']

    # Create the hiperparameters grid for the GridSearchCV
    param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
    }
    
    param_grid_rf = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 10, 20],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
    }

    # evaluate model Logistic Regression (adhd)
    lr_adhd = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='adhd')
    best_model_lr_adhd, f1_lr = lr_adhd.evaluate_with_gridsearch(
    base_model=LogisticRegression(max_iter=5000),
    param_grid=param_grid_lr,
    scoring='f1'
)

    # evaluate model Logistic Regression (sex_f)
    lr_sex_f = ModelEvaluation(X=df_train, y=labels[targets[1]], tag='sex_f')
    best_model_lr_sex_f, f1_lr = lr_sex_f.evaluate_with_gridsearch(
    base_model=LogisticRegression(max_iter=5000),
    param_grid=param_grid_lr,
    scoring='f1'
)
    
    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="adhd", sex_f_tag="sex_f")
    sub.to_submission(output_name='submission.csv')

    # Train and evaluate RandomForest for adhd
    rf_adhd = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='rf_adhd')
    best_model_rf_adhd, f1_rf = rf_adhd.evaluate_with_gridsearch(
    base_model=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    scoring='f1'
)

    # Train and evaluate RandomForest for sex_f
    rf_sex_f = ModelEvaluation(X=df_train, y=labels[targets[1]], tag='rf_sex_f')
    best_model_rf_sex_f, f1_rf = rf_sex_f.evaluate_with_gridsearch(
    base_model=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    scoring='f1'
)
    
    # Plots a tree of the forest
    graph_tree(best_model_rf_adhd, tag='rf_adhd')
    graph_tree(best_model_rf_sex_f, tag='rf_sex_f')

    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="rf_adhd", sex_f_tag="rf_sex_f")
    sub.to_submission(output_name='submission_rf.csv')

    # XGBoost model (for adhd)
    ev = ModelEvaluationXG(X=df_train, y=labels[targets[0]], tag='adhd')
    ev.evaluate_model()
    # XGBoost model (for sex_f)
    ev = ModelEvaluationXG(X=df_train, y=labels[targets[1]], tag='sex_f')
    ev.evaluate_model()

if __name__ == '__main__':
    main()