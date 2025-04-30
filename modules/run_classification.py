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
from model_params2 import Models



def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    # create a Dataset object
    df = Dataset(data_imputed = True, 
                 data_standarized = True, 
                 relevant_data = False)

    # train dataframe, test dataframe, y targets dataframe
    df_train, df_test, labels = df.process()

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

    #Initialized evaluation models.
    model = Models(X = df_train, 
                  y1 = labels[targets[0]],
                  y2 = labels[targets[1]], 
                  tag1 = 'adhd',
                  tag2 = 'sex_f', 
                  )
    
    #Evaluate model with Logistic Regression (adhd, sex_f).
    lr_adhd, lr_sex_f = model.log_regression(gridsearch = False,
                                                param_grid = param_grid_lr,
                                                scoring = 'f1',
                                                cv = 5,
                                           model_evaluation=True,
                                                max_iter = 5000,
                                                solver = 'lbfgs')
    
    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="adhd", sex_f_tag="sex_f")
    sub.to_submission(output_name='submission.csv')

    #Evaluate model with Random Forest (adhd, sex_f).
    rf_adhd, rf_sex_f = model.random_forest(gridsearch = False,
                                               param_grid = param_grid_rf,
                                               scoring = 'f1',
                                               cv = 5,
                                           model_evaluation=True,
                                               n_estimators = 100,
                                               criterion = "gini",
                                               max_depth = 10,
                                               random_state = 42,
                                               bootstrap = True)
   
    # Plots a tree of the forest
    graph_tree(rf_adhd, tag='rf_adhd')
    graph_tree(rf_sex_f, tag='rf_sex_f')

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