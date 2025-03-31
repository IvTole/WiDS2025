# Standar libraries
from datetime import datetime

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# External modules
from module_path import test_data_path, train_data_path, plots_data_path
from module_data import Dataset
from module_model import ModelEvaluation, ModelSubmission


def main():
    # get current date and time
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    df = Dataset()
    df_train, df_test, labels = df.load_data_frame_imputed() # train dataframe, test dataframe, y targets dataframe

    targets = ['ADHD_Outcome',  'Sex_F']

    # evaluate models (adhd)
    ev = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='adhd')
    ev.evaluate_model(LogisticRegression(solver='lbfgs', max_iter=5000))

    # evaluate models (sex_f)
    ev = ModelEvaluation(X=df_train, y=labels[targets[1]], tag='sex_f')
    ev.evaluate_model(LogisticRegression(solver='lbfgs', max_iter=5000))
    
    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="adhd", sex_f_tag="sex_f")
    sub.to_submission(output_name='submission.csv')

    # Train and evaluate RandomForest for adhd
    rf_adhd = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='rf_adhd')
    rf_adhd.evaluate_model(RandomForestClassifier(
        n_estimators=1000,
        criterion="gini",
        max_depth=10,
        random_state=42,
        bootstrap=True,
    ))

    # Train and evaluate RandomForest for sex_f
    rf_sex_f = ModelEvaluation(X=df_train, y=labels[targets[0]], tag='rf_sex_f')
    rf_sex_f.evaluate_model(RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        random_state=42,
        bootstrap=True,
    ))

    # prediction with test dataset
    sub = ModelSubmission(X=df_test, version=1, threshold=0.5, adhd_tag="rf_adhd", sex_f_tag="rf_sex_f")
    sub.to_submission(output_name='submission_rf.csv')

if __name__ == '__main__':
    main()