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
from module_model import ModelEvaluation, ModelSubmission


def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    # create a Dataset object
    df = Dataset()

    # train dataframe, test dataframe, y targets dataframe
    df_train, df_test, labels = df.load_data_frame_imputed()

    # Genera train dataframe, test dataframe, versión estandarizada para utilizar en futuros modelos 
    df_train_std, df_test_std = df.load_data_frame_standardized()

    # define array of target variables for the model
    targets = ['ADHD_Outcome',  'Sex_F']

    # Create the hiperparameters grid for the GridSearchCV
    param_grid_rf = {
    'n_estimators': [100, 300, 500, 750, 1000],
    'max_depth': [3, 5, 7, 10, 15, 20],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy', 'log_loss']
    }

    param_grid_lr = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'max_iter': [1000, 2500, 5000]
    }

    # Create grid for models
    grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')

    # evaluate models (adhd)
    ev = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='adhd')
    ev.evaluate_model(grid_lr)

    # evaluate models (sex_f)
    ev = ModelEvaluation(X=df_train, y=labels[targets[1]], tag='sex_f')
    ev.evaluate_model(grid_lr)
    
    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="adhd", sex_f_tag="sex_f")
    sub.to_submission(output_name='submission.csv')

    # Train and evaluate RandomForest for adhd
    rf_adhd = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='rf_adhd')
    rf_adhd.evaluate_model(grid_rf)

    # Train and evaluate RandomForest for sex_f
    rf_sex_f = ModelEvaluation(X=df_train, y=labels[targets[1]], tag='rf_sex_f')
    
    #sets the model 
    model_to_evaluate= RandomForestClassifier(grid_lr)
    rf_sex_f.evaluate_model(model=model_to_evaluate)
    
    # Plots a tree of the forest
    graph_tree(model_to_evaluate)

    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="rf_adhd", sex_f_tag="rf_sex_f")
    sub.to_submission(output_name='submission_rf.csv')

if __name__ == '__main__':
    main()