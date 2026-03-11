from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class HousePricePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders_ = {}
        self.num_imputer_    = SimpleImputer(strategy='median')
        self.cat_imputer_    = SimpleImputer(strategy='most_frequent')
        self.scaler_         = StandardScaler()

    def fit(self, X, y=None):
        X = X.copy()
        self.num_cols_ = X.select_dtypes(include=['int64','float64']).columns.tolist()
        self.cat_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        self.num_imputer_.fit(X[self.num_cols_])
        if self.cat_cols_:
            self.cat_imputer_.fit(X[self.cat_cols_])
        X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])
        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])
            for col in self.cat_cols_:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders_[col] = le
        X = self._encode(X)
        self.all_num_cols_ = X.columns.tolist()
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.num_cols_] = self.num_imputer_.transform(X[self.num_cols_])
        if self.cat_cols_:
            X[self.cat_cols_] = self.cat_imputer_.transform(X[self.cat_cols_])
        X = self._encode(X)
        X[self.all_num_cols_] = self.scaler_.transform(X[self.all_num_cols_])
        return X

    def _encode(self, X):
        X = X.copy()
        for col in self.cat_cols_:
            if col in X.columns:
                le = self.label_encoders_[col]
                X[col] = X[col].astype(str).apply(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                X[col] = le.transform(X[col])
        return X
