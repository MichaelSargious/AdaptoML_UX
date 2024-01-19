import numpy as np
import pandas as pd
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer



class Imputation:
    def __init__(self,
                 X,
                 X_Test=None,
                 ):
        self.X = X
        self.X_Test = X_Test

        self.factorization()

    def factorization(self):
        for col in self.X.columns:
            if self.X[col].dtype == "object" or self.X[col].dtype == "bool" or self.X[col].dtype == "category":
                self.X[col] = pd.factorize(self.X[col])[0]
                # self.X = self.X.drop([col], axis=1)

    def SimImp(self, missing_values=np.nan, strategy='mean', fill_value=None, verbose='deprecated', copy=True,
               add_indicator=False):

        imp = sklearn.impute.SimpleImputer(missing_values=missing_values, strategy=strategy, fill_value=fill_value,
                                           verbose=verbose, copy=copy, add_indicator=add_indicator)

        imp.fit(self.X)
        imputed_result_without_columns_names = imp.transform(self.X)
        imputed_result_with_columns_names = pd.DataFrame(imputed_result_without_columns_names, columns=self.X.columns)

        if self.X_Test:
            imputed_result_without_columns_names_test = imp.transform(self.X_Test)
            imputed_result_with_columns_names_test = pd.DataFrame(imputed_result_without_columns_names_test, columns=self.X_Test.columns)
            return imputed_result_with_columns_names, imputed_result_with_columns_names_test

        return imputed_result_with_columns_names

    def MultiImp(self, estimator=None, missing_values=np.nan, sample_posterior=False, max_iter=10, tol=0.001,
                 n_nearest_features=None, initial_strategy='mean', imputation_order='ascending', skip_complete=False,
                 min_value=-np.inf, max_value=np.inf, verbose=0, random_state=None, add_indicator=False):

        imp = sklearn.impute.IterativeImputer(estimator=estimator, missing_values=missing_values, sample_posterior=sample_posterior,
                                              max_iter=max_iter, tol=tol, n_nearest_features=n_nearest_features,
                                              initial_strategy=initial_strategy, imputation_order=imputation_order,
                                              skip_complete=skip_complete, min_value=min_value, max_value=max_value, verbose=verbose,
                                              random_state=random_state, add_indicator=add_indicator)

        imp.fit(self.X)
        imputed_result_without_columns_names = imp.transform(self.X)
        imputed_result_with_columns_names = pd.DataFrame(imputed_result_without_columns_names, columns=self.X.columns)

        if self.X_Test:
            imputed_result_without_columns_names_test = imp.transform(self.X_Test)
            imputed_result_with_columns_names_test = pd.DataFrame(imputed_result_without_columns_names_test, columns=self.X_Test.columns)
            return imputed_result_with_columns_names, imputed_result_with_columns_names_test

        return imputed_result_with_columns_names

    def KNNImp(self, missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean', copy=True,
               add_indicator=False):

        imp = sklearn.impute.KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors, weights=weights,
                                        metric=metric, copy=copy, add_indicator=add_indicator)

        imp.fit(self.X)
        imputed_result_without_columns_names = imp.transform(self.X)
        imputed_result_with_columns_names = pd.DataFrame(imputed_result_without_columns_names, columns=self.X.columns)

        if self.X_Test:
            imputed_result_without_columns_names_test = imp.transform(self.X_Test)
            imputed_result_with_columns_names_test = pd.DataFrame(imputed_result_without_columns_names_test, columns=self.X_Test.columns)
            return imputed_result_with_columns_names, imputed_result_with_columns_names_test

        return imputed_result_with_columns_names




