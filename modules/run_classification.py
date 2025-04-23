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
from module_graph import graph_tree
from module_model import ModelEvaluation, ModelSubmission
from model_params import get_lr, get_rf


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

    # define targets variables and his prefix for the model
    targets = {
        'ADHD_Outcome': 'adhd',
        'Sex_F': 'sex_f'
    }

    #List of models along with their parameters and their prefix.
    modelos = [
        ('Logistic Regression', get_lr, 'lr'),
        ('Random Forest', get_rf, 'rf'),
    ]

    #Evaluation loop for each model.
    submission_tags = {}
    for name, model_fn, prefix in modelos:
        print(f"\n Evaluando {name}")
        for col, tag in targets.items():
            model_tag = f"{prefix}_{tag}"
            evaluator = ModelEvaluation(X=df_train, y=labels[col], tag=model_tag)
            evaluator.evaluate_model(model_fn())
            submission_tags[tag] = model_tag

            #Graphic for models (Random Forest).
            if prefix == "rf" and tag == "sex_f":
                modelo_rf = model_fn()
                modelo_rf.fit(df_train, labels[col])
                graph_tree(modelo_rf)
        
        #Create the files that will be sent (submission section).
        sub = ModelSubmission(
            X=df_test,
            version=1,
            threshold=0.5,
            adhd_tag=submission_tags['adhd'],
            sex_f_tag=submission_tags['sex_f']
        )
        sub.to_submission(output_name=f"submission_{prefix}.csv")

if __name__ == '__main__':
    main()