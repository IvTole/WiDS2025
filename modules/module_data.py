# Standard libraries
import pandas as pd
import os

# Sklearn libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

# External libraries
from module_preprocessing import PreprocessingPipeline

from module_path import train_data_path, test_data_path, train_data_new_path

COL_EHQ_EHQ_TOTAL = "EHQ_EHQ_Total"
COL_COLORVISION_CV_SCORE = "ColorVision_CV_Score"
COL_APQ_P_APQ_P_CP = "APQ_P_APQ_P_CP"
COL_APQ_P_APQ_P_ID = "APQ_P_APQ_P_ID"
COL_APQ_P_APQ_P_INV = "APQ_P_APQ_P_INV"
COL_APQ_P_APQ_P_OPD = "APQ_P_APQ_P_OPD"
COL_APQ_P_APQ_P_PM = "APQ_P_APQ_P_PM"
COL_APQ_P_APQ_P_PP = "APQ_P_APQ_P_PP"
COL_SDQ_SDQ_CONDUCT_PROBLEMS = "SDQ_SDQ_Conduct_Problems"
COL_SDQ_SDQ_DIFFICULTIES_TOTAL = "SDQ_SDQ_Difficulties_Total"
COL_SDQ_SDQ_EMOTIONAL_PROBLEMS = "SDQ_SDQ_Emotional_Problems"
COL_SDQ_SDQ_EXTERNALIZING = "SDQ_SDQ_Externalizing"
COL_SDQ_SDQ_GENERATING_IMPACT = "SDQ_SDQ_Generating_Impact"
COL_SDQ_SDQ_HYPERACTIVITY = "SDQ_SDQ_Hyperactivity"
COL_SDQ_SDQ_INTERNALIZING = "SDQ_SDQ_Internalizing"
COL_SDQ_SDQ_PEER_PROBLEMS = "SDQ_SDQ_Peer_Problems"
COL_SDQ_SDQ_PROSOCIAL = "SDQ_SDQ_Prosocial"
COL_MRI_TRACK_AGE_AT_SCAN = "MRI_Track_Age_at_Scan"
COL_BASIC_DEMOS_ENROLL_YEAR = "Basic_Demos_Enroll_Year"
COL_BASIC_DEMOS_STUDY_SITE = "Basic_Demos_Study_Site"
COL_PREINT_DEMOS_FAM_CHILD_ETHNICITY = "PreInt_Demos_Fam_Child_Ethnicity"
COL_PREINT_DEMOS_FAM_CHILD_RACE = "PreInt_Demos_Fam_Child_Race"
COL_MRI_TRACK_SCAN_LOCATION = "MRI_Track_Scan_Location"
COL_BARRATT_BARRATT_P1_EDU = "Barratt_Barratt_P1_Edu"
COL_BARRATT_BARRATT_P1_OCC = "Barratt_Barratt_P1_Occ"
COL_BARRATT_BARRATT_P2_EDU = "Barratt_Barratt_P2_Edu"
COL_BARRATT_BARRATT_P2_OCC = "Barratt_Barratt_P2_Occ"


class Dataset:

    def __init__(self, num_samples: int = None, random_seed: int = 42):
        """
        :param num_samples: the number of samples to draw from the data frame; if None, use all samples
        :param random_seed: the random seed to use when sampling data points
        """

        self.num_samples = num_samples
        self.random_seed = random_seed
    
    def load_data_frame(self) -> pd.DataFrame:
        """
        :return: the full data frame for this dataset for train features, test features, and training labels

        Note: Null values are dropped and invoice date variable type is changed to timestamp
        """

        #train_path = train_data_path()
        train_new_path = train_data_new_path()
        test_path = test_data_path()
        
        # old datasets for trainning
        #train_q = pd.read_excel(os.path.join(train_path,"TRAIN_QUANTITATIVE_METADATA.xlsx"))
        #train_c = pd.read_excel(os.path.join(train_path,"TRAIN_CATEGORICAL_METADATA.xlsx"))

        #train_combined = pd.merge(train_q, train_c, on="participant_id", how="left").set_index("participant_id")
        
        # new datasets for trainning
        train_new_q = pd.read_excel(os.path.join(train_new_path,"TRAIN_QUANTITATIVE_METADATA_new.xlsx"))
        train_new_c = pd.read_excel(os.path.join(train_new_path,"TRAIN_CATEGORICAL_METADATA_new.xlsx"))
        
        train_new_combined = pd.merge(train_new_q, train_new_c, on="participant_id", how="left").set_index("participant_id").sort_index()

        # datasets for testing
        test_q = pd.read_excel(os.path.join(test_path,"TEST_QUANTITATIVE_METADATA.xlsx"))
        test_c = pd.read_excel(os.path.join(test_path,"TEST_CATEGORICAL.xlsx"))        
                
        test_combined = pd.merge(test_q, test_c, on="participant_id", how="left").set_index("participant_id")

        labels = pd.read_excel(os.path.join(train_new_path,"TRAINING_SOLUTIONS.xlsx")).set_index("participant_id").sort_index()
        assert all(train_new_combined.index == labels.index), "Label IDs don't match train IDs"

        # Sample
        if self.num_samples is not None:
            train_new_combined = train_new_combined.sample(self.num_samples, random_state=self.random_seed)   

        return train_new_combined, test_combined, labels
    
    def load_data_frame_imputed(self):
        train_combined, test_combined, labels = self.load_data_frame()

        impute_train = Imputer(df=train_combined)
        impute_test = Imputer(df=test_combined)

        n_neighbors = 5
        train_combined_imputed = impute_train.imputer_knn(n_neighbors=n_neighbors)
        test_combined_imputed = impute_test.imputer_knn(n_neighbors=n_neighbors)
        print(f'NaN values processed for every dataset by kNNImputer algorithm considering {n_neighbors} neighbors.')

        return train_combined_imputed, test_combined_imputed, labels
    
    def load_data_frame_standardized(self):
        train_data, test_data, labels = self.load_data_frame_imputed()

        # Inicializar preprocesador
        preprocessor = Preprocessor(method="standard")

        # Ajustar y transformar train
        train_standardized = preprocessor.fit_transform(train_data)

        # Convertir a DataFrame
        train_standardized = pd.DataFrame(train_standardized, columns=train_data.columns, index=train_data.index)

        # Transformar test con los mismos parámetros del train
        test_standardized = preprocessor.transform(test_data)

        # Convertir a DataFrame
        test_standardized = pd.DataFrame(test_standardized, columns=test_data.columns, index=test_data.index)

        # Mostrar una vista previa de los datos preprocesados
        print(train_standardized.head())
        print(test_standardized.head())

        return train_standardized, test_standardized


class Imputer():

    def __init__(self, df: pd.DataFrame, n_neighbors:int=5):
        self.df = df

    def imputer_knn(self, n_neighbors):

        df = self.df.copy()

        imputer = KNNImputer(n_neighbors=n_neighbors)
        column_names = df.columns
        index_ = df.index
        df = imputer.fit_transform(df)
        df = pd.DataFrame(df, columns=column_names)
        df.index = index_
        
        return df


class Preprocessor:
    def __init__(self, method: str = "standard"):
        """
        :param method: Método de estandarización ("standard" para StandardScaler, "minmax" para MinMaxScaler).
        """
        self.method = method
        self.scaler = StandardScaler() if self.method == "standard" else MinMaxScaler()

    def fit(self, df: pd.DataFrame):
        """
        Ajusta el scaler con las columnas numéricas del train set.
        """
        num_cols = df.select_dtypes(include=["number"]).columns
        self.scaler.fit(df[num_cols])

    def transform(self, df: pd.DataFrame):
        """
        Transforma el dataset usando el scaler ajustado.
        """
        df_copy = df.copy()
        num_cols = df.select_dtypes(include=["number"]).columns
        df_copy[num_cols] = self.scaler.transform(df_copy[num_cols])
        return df_copy

    def fit_transform(self, df: pd.DataFrame):
        """
        Ajusta y transforma el dataset (solo debe usarse en el train set).
        """
        self.fit(df)
        return self.transform(df)
    


