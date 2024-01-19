import pickle
import warnings
import numpy as np
import pandas as pd
import Utilities as U
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import chi2
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

warnings.filterwarnings("ignore")


# compute the empirical rule in a dict
def criteria_dict_construction(mean, std):
    criteria_dict = {
        68: mean + (1 * std),
        95: mean + (2 * std),
        99: mean + (3 * std)
    }
    return criteria_dict


class FeatureSelection_Methods:
    def __init__(self, X_Train,
                 X_Val,
                 Y_Train=None,
                 Y_Val=None,
                 CI=95,
                 k=10,
                 method_selection=None,
                 save_op_csv=False,
                 save_op_pickle=False,
                 save_op_json=False,
                 metric="accuracy",
                 task_type=None
                 ):

        if method_selection is None:
            method_selection = ['missing_ratio', 'chi_sq', 'fisher', 'permutation_importance',
                                'correlation_coefficient', 'low_variance', 'MAD', 'mutual_information', 'lasso',
                                "elastic_net", "random_forest", "step_forward_selection", "step_backward_selection",
                                'bi-directional_selection']
            # 'exhaustive_selection']

        if isinstance(X_Train, np.ndarray):
            self.X_Train = pd.DataFrame(X_Train)
        else:
            self.X_Train = X_Train
        if isinstance(Y_Train, np.ndarray):
            self.Y_Train = pd.DataFrame(Y_Train)
        else:
            self.Y_Train = Y_Train
        if isinstance(X_Val, np.ndarray):
            self.X_Val = pd.DataFrame(X_Val)
        else:
            self.X_Val = X_Val
        if isinstance(Y_Val, np.ndarray):
            self.Y_Val = pd.DataFrame(Y_Val)
        else:
            self.Y_Val = Y_Val

        if len(self.X_Train.columns) < k:
            self.k = len(self.X_Train.columns)
        else:
            self.k = k

        self.CI = CI
        self.method_selection = method_selection
        self.save_op_csv = save_op_csv
        self.save_op_pickle = save_op_pickle
        self.save_op_json = save_op_json

        self.selected_features_dataframes = []
        self.output_dict = {}
        self.metric = metric
        self.task_type = task_type

        if "error" in self.metric:
            self.best_score = 10000
        else:
            self.best_score = -1

    def selection_methods(self):

        if 'exhaustive_selection' in self.method_selection:
            print('Warning Message: exhaustive_selection might take some time.')
        if 'chi_sq' in self.method_selection and (self.X_Train < 0).values.any() and self.Y_Train is not None:
            print('Warning : chi-sq can not be used because there are some negative values.')

        assert self.CI in [68, 95, 99], "Confidence interval (CI) must be specified, can be 90, 95 or 99."
        assert all(method in ['missing_ratio', 'chi_sq', 'fisher', 'permutation_importance', 'correlation_coefficient',
                              'low_variance', 'MAD', 'mutual_information', "lasso", "elastic_net", "random_forest",
                              "step_forward_selection", "step_backward_selection", 'bi-directional_selection',
                              'exhaustive_selection']
                   for method in
                   self.method_selection), "method_selection must be a list of 'missing_ratio', 'chi_sq', 'fisher', " \
                                           "'permutation_importance', 'correlation_coefficient', 'low_variance', " \
                                           "'MAD', 'mutual_information', 'lasso', 'elastic_net', 'random_forest', " \
                                           "'step_forward_selection', 'step_backward_selection', " \
                                           "'bi-directional_selection' or 'exhaustive_selection'"

        original_columns_names = self.X_Train.columns

        for method in self.method_selection:

            selected_variables = []

            # missing_ratio :
            if method == "missing_ratio":
                missing_ratio = self.X_Train.isnull().sum() / len(self.X_Train) * 100
                criteria_dict = criteria_dict_construction(missing_ratio.mean(), missing_ratio.std())

                for i in range(self.X_Train.columns.shape[0]):
                    if missing_ratio[i] <= criteria_dict[self.CI]:
                        selected_variables.append(original_columns_names[i])

                if not selected_variables:
                    print("Warning : " + method + " method did not deliver any results.")
                else:
                    df_subset = self.X_Train[selected_variables]

                    columns_two_level_names = []
                    for feature_name in selected_variables:
                        columns_two_level_names.append((method, feature_name))
                    df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                    df_subset = df_subset.reset_index()

                    self.selected_features_dataframes.append(df_subset)
                    self.output_dict[method] = selected_variables

            if method == "low_variance":
                self.X_Train = self.X_Train.apply(
                    lambda iterator: ((iterator - iterator.mean()) / iterator.std()).round(8))
                variance_table = self.X_Train.var()
                criteria_dict = criteria_dict_construction(variance_table.mean(), variance_table.std())

                for i in range(self.X_Train.columns.shape[0]):
                    if variance_table[i] <= criteria_dict[self.CI]:
                        selected_variables.append(original_columns_names[i])

                if not selected_variables:
                    print("Warning : " + method + " method did not deliver any results.")
                else:
                    df_subset = self.X_Train[selected_variables]

                    columns_two_level_names = []
                    for feature_name in selected_variables:
                        columns_two_level_names.append((method, feature_name))
                    df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                    df_subset = df_subset.reset_index()

                    self.selected_features_dataframes.append(df_subset)
                    self.output_dict[method] = selected_variables

            if method == "MAD":
                MDA = np.sum(np.abs(self.X_Train - np.mean(self.X_Train, axis=0)), axis=0) / \
                      self.X_Train.shape[0]
                criteria_dict = criteria_dict_construction(MDA.mean(), MDA.std())

                for i in range(self.X_Train.columns.shape[0]):
                    if MDA[i] <= criteria_dict[self.CI]:
                        selected_variables.append(original_columns_names[i])

                if not selected_variables:
                    print("Warning : " + method + " method did not deliver any results.")
                else:
                    df_subset = self.X_Train[selected_variables]

                    columns_two_level_names = []
                    for feature_name in selected_variables:
                        columns_two_level_names.append((method, feature_name))
                    df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                    df_subset = df_subset.reset_index()

                    self.selected_features_dataframes.append(df_subset)
                    self.output_dict[method] = selected_variables

            if self.Y_Train is not None:
                if method == "chi_sq":
                    selected_variables = []
                    if (self.X_Train < 0).values.any():
                        continue

                    self.X_Train = self.X_Train.fillna(0)
                    chi2_selector = SelectKBest(chi2, k=self.k)
                    X_kbest = chi2_selector.fit_transform(self.X_Train, self.Y_Train)

                    for col in self.X_Train.columns:
                        for i in range(X_kbest.shape[1]):
                            if np.array_equal(self.X_Train[col], X_kbest[:, i]):
                                selected_variables.append(col)

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "fisher":
                    idx = fisher_score.fisher_score(self.X_Train.to_numpy(), self.Y_Train.to_numpy(), mode='rank')

                    small_range = self.k
                    if self.k > len(self.X_Train.columns):
                        small_range = len(self.X_Train.columns)

                    for i in range(small_range):
                        selected_variables.append(original_columns_names[idx[i]])

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "permutation_importance":
                    X_train, X_val, y_train, y_val = train_test_split(self.X_Train, self.Y_Train, random_state=0)
                    model = Ridge(alpha=1e-2).fit(X_train, y_train)
                    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)

                    for i in r.importances_mean.argsort()[::-1]:
                        criteria_dict = criteria_dict_construction(r.importances_mean[i], r.importances_std[i])
                        if criteria_dict[self.CI] > 0:
                            selected_variables.append(original_columns_names[i])

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "correlation_coefficient":
                    temp_df = self.X_Train.copy()
                    temp_df["target"] = self.Y_Train
                    cor = temp_df.corr()
                    cor_target = abs(cor["target"])
                    selected_variables = cor_target[cor_target > 0.5].index.values.tolist()
                    selected_variables = selected_variables.remove('target')

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables.remove('target')]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables.remove('target')

                if method == "mutual_information" and self.task_type == "Classification":
                    mutual_info_selector = SelectKBest(mutual_info_classif, k=self.k)
                    mutual_info_selector.fit_transform(self.X_Train, self.Y_Train)
                    selected_variables = self.X_Train.columns[mutual_info_selector.get_support()].tolist()

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "lasso" and self.task_type == "Classification":
                    skf = StratifiedKFold(n_splits=10)
                    lasso = LassoCV(cv=skf, random_state=42).fit(self.X_Train, self.Y_Train)
                    selected_variables = list(self.X_Train.columns[np.where(lasso.coef_ != 0)[0]])

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "elastic_net":
                    model = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(self.X_Train, self.Y_Train)
                    selected_variables = list(self.X_Train.columns[np.where(model.coef_ != 0)[0]])

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "random_forest":
                    rf = RandomForestRegressor(n_estimators=100).fit(self.X_Train, self.Y_Train)
                    sorted_idx = rf.feature_importances_.argsort().tolist()
                    selected_variables = self.X_Train.columns[sorted_idx[0:self.k]].tolist()

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "step_forward_selection":
                    sfs = SFS(LinearRegression(), k_features=self.k, forward=True, floating=False, scoring='r2', cv=0)
                    sfs.fit(self.X_Train, self.Y_Train)
                    selected_variables = list(sfs.k_feature_names_)

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == "step_backward_selection":
                    sbs = SFS(LinearRegression(), k_features=self.k, forward=False, floating=False, scoring='r2', cv=0)
                    sbs.fit(self.X_Train, self.Y_Train)
                    selected_variables = list(sbs.k_feature_names_)

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == 'bi-directional_selection':
                    sffs = SFS(LinearRegression(), k_features=(self.k, self.k), forward=True, floating=True, cv=0)
                    sffs.fit(self.X_Train, self.Y_Train)
                    selected_variables = list(sffs.k_feature_names_)

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

                if method == 'exhaustive_selection':
                    efs = EFS(LinearRegression(), min_features=1, max_features=self.k, scoring='accuracy',
                              print_progress=True, cv=5).fit(self.X_Train, self.Y_Train)
                    selected_variables = list(efs.k_feature_names_)

                    if not selected_variables:
                        print("Warning : " + method + " method did not deliver any results.")
                    else:
                        df_subset = self.X_Train[selected_variables]

                        columns_two_level_names = []
                        for feature_name in selected_variables:
                            columns_two_level_names.append((method, feature_name))
                        df_subset.columns = pd.MultiIndex.from_tuples(columns_two_level_names)
                        df_subset = df_subset.reset_index()

                        self.selected_features_dataframes.append(df_subset)
                        self.output_dict[method] = selected_variables

        if self.Y_Train is None:
            print("Warning : Other methods can not be used because there is not Y specified.")

        return self.selected_features_dataframes

    def grid_search_FS(self):
        if self.Y_Val is None:
            print(
                "grid_search can not be used because there is no label (Y) specified. Normal selection methods will be performed without grid search.")
            return self.selection_methods()

        # assert self.output_dict is not None, "selection_methods() function must computed first."
        if not self.output_dict:
            self.selection_methods()

        all_SF_with_scores_df = pd.DataFrame()
        not_existing_methods = []

        name_of_best_score = ""
        selected_features_with_best_score = []
        dataframe_of_best_score = pd.DataFrame()

        for method in self.method_selection:
            if method not in self.output_dict.keys():
                not_existing_methods.append(method)
                features_subset = self.X_Val.columns
            else:
                features_subset = list(self.output_dict[method])
            df_subset = self.X_Val[features_subset]
            eva_dict = U.forest_test(df_subset, self.Y_Val, self.task_type)
            result = U.get_score(eva_dict, self.metric)
            all_SF_with_scores_df[method + "_with_score_" + str(result)] = [features_subset]

            if self.task_type == "Classification":
                if self.best_score < result:
                    self.best_score = result
                    name_of_best_score = method
                    selected_features_with_best_score = features_subset
            else:
                if "error" in self.metric:
                    if self.best_score > result:
                        self.best_score = result
                        name_of_best_score = method
                        selected_features_with_best_score = features_subset
                else:
                    if self.best_score < result:
                        self.best_score = result
                        name_of_best_score = method
                        selected_features_with_best_score = features_subset

        dataframe_of_best_score[name_of_best_score + "_with_score_" + str(self.best_score)] = [
            selected_features_with_best_score]

        print("Warning: the following methods " + str(not_existing_methods) + " did not deliver any selected subsets, "
                                                                              "that's why the whole features are"
                                                                              " considered in the evaluation")

        if self.save_op_csv:
            dataframe_of_best_score.to_csv(
                "results/best_selected_features_are_from_" + name_of_best_score + "_with_score_" +
                str(self.best_score) + ".csv", sep=";")
            all_SF_with_scores_df.to_csv("results/all_selected_features_with_scores_in_one.csv", sep=";")

        final_dict_with_all_information = {
            "original_training_features": self.X_Train,
            "original_training_target": self.Y_Train,
            "original_validation_features": self.X_Val,
            "original_validation_target": self.Y_Val,
            "all_selected_features": all_SF_with_scores_df,
            "name_of_best_score": name_of_best_score,
            "features_name_with_best_score": selected_features_with_best_score,
            "best_selected_features_from_training_data": self.X_Train[selected_features_with_best_score],
            "best_selected_features_from_validation_data": self.X_Val[selected_features_with_best_score]
        }

        if self.save_op_pickle:
            with open('results/final_dict_with_all_information.pickle', 'wb') as fp:
                pickle.dump(final_dict_with_all_information, fp)
            fp.close()

        return final_dict_with_all_information
