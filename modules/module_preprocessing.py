from sklearn.preprocessing import StandardScaler, MinMaxScaler

class PreprocessingPipeline:
    def __init__(self, method="standard"):
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)