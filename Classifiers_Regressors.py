import pickle
import os
import Utilities as U
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import naive_bayes
from skmultiflow import lazy
from skmultiflow import trees
from sklearn.metrics import classification_report


def adapting_models(task_type, metric_type, X_train, Y_train, X_test, Y_test, X_val, Y_val, per_col_name,
                    per_col_test, test_size, filePath_adapted_previous_info):
    global best_model
    all_persons_with_all_metrics = {}
    all_others_with_all_metrics = {}
    previous_persons_avg_dict = {}
    previous_others_avg_dict = {}
    all_persons_to_plot = {}
    all_others_to_plot = {}
    session_nr = 1

    clf = Classifiers_Regressors(task_type, metric_type)

    if not filePath_adapted_previous_info:
        # fitting UBM and evaluating it with test data
        clf.myFit(X_train, Y_train)

        # info about the best model (UBM)
        # UBM evaluation report on test data
        _, UBM_model_eva = clf.best_model_selection(X_val, Y_val, X_test, Y_test)

        # adding UBM weighted avg and acc to the reports
        if task_type == "Classification":
            temp_dict = UBM_model_eva[1]["weighted avg"]
            temp_dict["accuracy"] = UBM_model_eva[1]["accuracy"]

            all_persons_with_all_metrics["UBM"] = temp_dict
            all_others_with_all_metrics["UBM"] = temp_dict

            all_persons_to_plot["UBM"] = temp_dict
            all_others_to_plot["UBM"] = temp_dict

            previous_persons_avg_dict["UBM"] = temp_dict
            previous_others_avg_dict["UBM"] = temp_dict
        else:
            all_persons_with_all_metrics["UBM"] = list(UBM_model_eva[1].values())[0]
            all_others_with_all_metrics["UBM"] = list(UBM_model_eva[1].values())[0]

            all_persons_to_plot["UBM"] = list(UBM_model_eva[1].values())[0]
            all_others_to_plot["UBM"] = list(UBM_model_eva[1].values())[0]

            previous_persons_avg_dict["UBM"] = list(UBM_model_eva[1].values())[0]
            previous_others_avg_dict["UBM"] = list(UBM_model_eva[1].values())[0]

    else:
        with open(filePath_adapted_previous_info, 'rb') as fp:
            previous_adapted_data_dict = pickle.load(fp)
        fp.close()

        all_persons_with_all_metrics = previous_adapted_data_dict["all_persons_with_all_metrics"]
        all_others_with_all_metrics = previous_adapted_data_dict["all_others_with_all_metrics"]
        previous_persons_avg_dict = previous_adapted_data_dict["previous_persons_avg_dict"]
        previous_others_avg_dict = previous_adapted_data_dict["previous_others_avg_dict"]
        UBM_model_eva = previous_adapted_data_dict["UBM_model_eva"]
        clf.best_model = previous_adapted_data_dict["UBM"]
        clf.fit_flag = True
        session_nr = previous_adapted_data_dict["session_nr"] + 1

        all_persons_to_plot["UBM"] = all_persons_with_all_metrics["UBM"]
        all_others_to_plot["UBM"] = all_others_with_all_metrics["UBM"]

    X_test["label"] = Y_test
    X_test[per_col_name] = per_col_test
    groups = X_test.groupby(per_col_name)
    df_list = {}

    # splitting and order all runs from test data in a dictionary as (test, train).
    if not filePath_adapted_previous_info:
        run_nr = 1
    else:
        run_nr = len(previous_adapted_data_dict["all_persons_with_all_metrics"]) - 1

    for i, g in enumerate(groups, run_nr):
        test_ratio = round(test_size * len(g[1]))
        df_list[per_col_name + "_" + str(g[1][per_col_name].iloc[0])] = (
        g[1][:test_ratio], g[1][test_ratio:])  # (test, train)

    for person_name, person_test_train_data in df_list.items():

        # setting UBM as default model
        clf.models = clf.best_model

        # increment one run.
        y_inc = person_test_train_data[1]["label"]
        x_inc = person_test_train_data[1].drop(["label", per_col_name], axis=1)
        clf.myPartial_fit(x_inc, y_inc)

        # getting first report by testing with the test half of that run.
        y_ts = person_test_train_data[0]["label"]
        x_ts = person_test_train_data[0].drop(["label", "user_id"], axis=1)
        run_eva = list(clf.myEvaluation_report(x_ts, y_ts, clf.models).values())[0]
        if task_type == "Classification":
            temp_dict = run_eva["weighted avg"]
            temp_dict["accuracy"] = run_eva["accuracy"]
            all_persons_with_all_metrics[person_name] = temp_dict
            all_persons_to_plot[person_name] = temp_dict
        else:
            all_persons_with_all_metrics[person_name] = run_eva
            all_persons_to_plot[person_name] = run_eva

        # combining all the test data halves of the other runs.
        temp_other_test_dfs = [value[0] for key, value in df_list.items() if key != person_name]
        temp_other_test_dfs = pd.concat(temp_other_test_dfs)

        # getting second report by evaluating all others halves of other runs
        y_tss = temp_other_test_dfs["label"]
        x_tss = temp_other_test_dfs.drop(["label", "user_id"], axis=1)
        others_eva = list(clf.myEvaluation_report(x_tss, y_tss, clf.models).values())[0]
        if task_type == "Classification":
            temp_dict = others_eva["weighted avg"]
            temp_dict["accuracy"] = others_eva["accuracy"]
            all_others_with_all_metrics["others_except_" + person_name] = temp_dict
            all_others_to_plot["others_except_" + person_name] = temp_dict
        else:
            all_others_with_all_metrics["others_except_" + person_name] = others_eva
            all_others_to_plot["others_except_" + person_name] = others_eva


    person_df, other_df = U.print_csv_files(all_persons_with_all_metrics, all_others_with_all_metrics)

    person_df.to_csv(
        "results/all_adapted_entities_in_all_performed_session_with_" + list(clf.best_model.keys())[0] + "_df.csv")
    other_df.to_csv(
        "results/all_non_entities_persons_in_all_performed_session_with_" + list(clf.best_model.keys())[0] + "_df.csv")

    U.plot_all_metrics(all_persons_to_plot, all_others_to_plot, task_type, session_nr)

    previous_persons_avg_dict["session_" + str(session_nr)] = U.reports_average(list(all_persons_to_plot.values())[1:])
    previous_others_avg_dict["session_" + str(session_nr)] = U.reports_average(list(all_others_to_plot.values())[1:])

    U.plot_sessions_graphs(previous_persons_avg_dict, previous_others_avg_dict, task_type)

    previous_adapted_data_dict = {
        "all_persons_with_all_metrics": all_persons_with_all_metrics,
        "all_others_with_all_metrics": all_others_with_all_metrics,
        "previous_persons_avg_dict": previous_persons_avg_dict,
        "previous_others_avg_dict": previous_others_avg_dict,
        "UBM_model_eva": UBM_model_eva,
        "UBM": clf.best_model,
        "session_nr": session_nr
    }

    with open("results/adapted_data_file_in_session_" + str(session_nr) + ".pickle", 'wb') as fp:
        pickle.dump(previous_adapted_data_dict, fp)
    fp.close()


class Classifiers_Regressors:
    def __init__(self, models_type, metric_type):

        self.fit_flag = False
        self.models_type = models_type
        self.metric_type = metric_type
        self.best_model = {}
        if "error" in self.metric_type:
            self.best_score = 10000
        else:
            self.best_score = -1

        if self.models_type == "Classification":
            self.models = {
                "ExtremelyFastDecisionTreeClassifier": trees.ExtremelyFastDecisionTreeClassifier(),
                "HoeffdingTreeClassifier": trees.HoeffdingTreeClassifier(),
                "Lazy_KNNClassifier": lazy.KNNClassifier(),
                "Lazy_SAMKNNClassifier": lazy.SAMKNNClassifier(),
                "Lazy_KNNADWINClassifier": lazy.KNNADWINClassifier(),
                "Linear_Perceptron": linear_model.Perceptron(),
                "Linear_PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier(),
                "Linear_SGD": linear_model.SGDClassifier(),
                "NaiveBayes_BernoulliNB": naive_bayes.BernoulliNB(),
                "NaiveBayes_MultinominalNB": naive_bayes.MultinomialNB()
            }
        elif self.models_type == "Regression":
            self.models = {
                "Lazy_KNNRegressor": lazy.KNNRegressor(),
                "Linear_SGDRegressor": linear_model.SGDRegressor(),
                "Linear_PassiveAggressiveRegressor": linear_model.PassiveAggressiveRegressor()
            }

    def myFit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()

        not_allowed_clf = []
        for name, model in self.models.items():
            try:
                model.fit(X, y)
            except Exception as exc:
                print('{err}. Therefore '.format(err=exc) + name + " can not be used")
                not_allowed_clf.append(name)

        if not_allowed_clf:
            for model in not_allowed_clf:
                self.models.pop(model, None)

        self.fit_flag = True

    def myPredict(self, x):
        assert self.fit_flag, "You must fit the models first with myFit() function."

        if not isinstance(x, np.ndarray):
            x = x.to_numpy()

        prediction_dict = {}

        for name, model in self.models.items():
            try:
                prediction_dict["prediction_of_" + name] = model.predict(x)
            except Exception as exc:
                print('{err}. Therefore '.format(err=exc) + name + " can not be used")

        return prediction_dict

    def end_to_end_partial_fit(self, X_inc, Y_inc, X_val, Y_val, X_test, Y_test, filePath_incremental_previous_data):
        global temp_dict
        eva_dict = {}

        # getting the model evaluation before incrementation
        if filePath_incremental_previous_data is None:
            assert self.fit_flag, "You must fit the models first with myFit() function."
            session_nr = 0

            _, UBM_model_eva = self.best_model_selection(X_val, Y_val, X_test, Y_test)

            # adding UBM weighted avg and acc to the reports
            if self.models_type == "Classification":
                temp_dict = UBM_model_eva[1]["weighted avg"]
                temp_dict["accuracy"] = UBM_model_eva[1]["accuracy"]

            else:
                temp_dict = list(UBM_model_eva[1].values())[0]

            temp_dict["kemker_metric"] = temp_dict[self.metric_type]
            eva_dict["base_model"] = temp_dict
        else:
            with open(filePath_incremental_previous_data, 'rb') as fp:
                previous_incremental_data_dict = pickle.load(fp)
            fp.close()

            self.best_model = previous_incremental_data_dict["best_model"]
            eva_dict = previous_incremental_data_dict["eva_list"]
            session_nr = previous_incremental_data_dict["session_nr"]

        # incrementing the model
        if not isinstance(X_inc, np.ndarray):
            X_inc = X_inc.to_numpy()
        if not isinstance(Y_inc, np.ndarray):
            Y_inc = Y_inc.to_numpy()

        for name, model in self.best_model.items():
            try:
                model.partial_fit(X_inc, Y_inc)
                session_nr += 1
            except Exception as exc:
                print('{err}. Therefore '.format(err=exc) + name + " can not be incremented with this new data.")

        # evaluating the model after incrementation
        if not isinstance(X_test, np.ndarray):
            X_test = X_test.to_numpy()
        if not isinstance(Y_test, np.ndarray):
            Y_test = Y_test.to_numpy()

        if self.models_type == "Classification":
            for name, model in self.best_model.items():
                try:
                    Y_pred = model.predict(X_test)
                    class_report = classification_report(Y_test, Y_pred, output_dict=True)
                    temp_dict = class_report["weighted avg"]
                    temp_dict["accuracy"] = class_report["accuracy"]

                except Exception as exc:
                    print('{err}. Therefore '.format(err=exc) + name + " can not give an evaluation report")
        else:
            for name, model in self.best_model.items():
                try:
                    Y_pred = model.predict(X_test)
                    temp_dict = U.regressor_evaluation(Y_test, Y_pred)

                except Exception as exc:
                    print('{err}. Therefore '.format(err=exc) + str(name) + " can not give an evaluation report")

        temp_dict["kemker_metric"] = np.NAN
        eva_dict["session_" + str(session_nr)] = temp_dict

        eva_dict = U.compute_kemker_metric(eva_dict, session_nr, self.metric_type)

        eva_df = U.print_csv_increment(eva_dict)
        eva_df.to_csv("results/all_incremented_sessions_with_" + list(self.best_model.keys())[0] + "_df.csv")

        U.plot_incremental_graphs(eva_dict, self.models_type)

        previous_incremental_data_dict = {
            "best_model": self.best_model,
            "eva_list": eva_dict,
            "session_nr": session_nr,
        }

        with open("results/previous_incremented_data_with_all_session.pickle", 'wb') as fp:
            pickle.dump(previous_incremental_data_dict, fp)
        fp.close()

    def myPartial_fit(self, X_inc, Y_inc):
        assert self.fit_flag, "You must fit the models first with myFit() function."

        if not isinstance(X_inc, np.ndarray):
            X_inc = X_inc.to_numpy()
        if not isinstance(Y_inc, np.ndarray):
            Y_inc = Y_inc.to_numpy()

        for name, model in self.models.items():
            try:
                model.partial_fit(X_inc, Y_inc)
            except Exception as exc:
                print('{err}. Therefore '.format(err=exc) + name + " can not be incremented with this new data.")

    def myEvaluation_report(self, X_test, Y_test, models):
        global Y_pred
        assert self.fit_flag, "You must fit the models first with myFit() function."

        if not isinstance(X_test, np.ndarray):
            X_test = X_test.to_numpy()
        if not isinstance(Y_test, np.ndarray):
            Y_test = Y_test.to_numpy()

        eva_dict = {}

        if self.models_type == "Classification":
            for name, model in models.items():
                try:
                    Y_pred = model.predict(X_test)
                    eva_dict[name] = classification_report(Y_test, Y_pred, output_dict=True)

                except Exception as exc:
                    print('{err}. Therefore '.format(err=exc) + name + " can not give an evaluation report")
        else:
            for name, model in models.items():
                try:
                    Y_pred = model.predict(X_test)
                    eva_dict[name] = U.regressor_evaluation(Y_test, Y_pred)

                except Exception as exc:
                    print('{err}. Therefore '.format(err=exc) + str(name) + " can not give an evaluation report")

        return eva_dict

    def saving_models(self, save_all_in_one_file=False):
        if "saved_models" not in os.listdir():
            os.mkdir("saved_models")

        if save_all_in_one_file:
            pickle.dump(self.models, open("saved_models/all_models_in_dict.pickle", 'wb'))
            print("All models have been saved in one file as a dictionary.")
        else:
            for name, clf in self.models.items():
                pickle.dump(clf, open("saved_models/" + name + ".pickle", 'wb'))
            print("each model has been saved in separate file.")

    def loading_models(self, files_paths=None):
        assert files_paths is not None, "You must assign the path of the file of all models that needed to be loaded " \
                                        "to \"files_paths\". All models must be saved on that file as a dictionary." \
                                        " However, you can assign all the paths of files of all models as a list to" \
                                        " \"files_paths\", if each model saved in separate file."

        if len(files_paths) > 1:
            loaded_models = {}
            for file in files_paths:
                loaded_models[file] = pickle.load(open(file, 'rb'))
            self.models = loaded_models
        else:
            for file in files_paths:
                self.models = pickle.load(open(file, 'rb'))

        self.fit_flag = True

    def best_model_selection(self, X_val, Y_val, X_test, Y_test):
        assert self.fit_flag, "You must fit the models first with myFit() function."

        if not isinstance(X_val, np.ndarray):
            X_val = X_val.to_numpy()
        if not isinstance(Y_val, np.ndarray):
            Y_val = Y_val.to_numpy()

        eva_dict = {}
        best_name = None
        best_model = None

        all_models_score_df = pd.DataFrame()

        if self.models_type == "Classification":
            for name, model in self.models.items():
                try:
                    Y_pred = model.predict(X_val)
                    eva_dict[name] = classification_report(Y_val, Y_pred, output_dict=True)
                    result = U.get_score(eva_dict[name], self.metric_type)

                    all_models_score_df[name + "_score"] = [result]

                    if result > self.best_score:
                        self.best_score = result
                        best_name = name
                        best_model = model


                except Exception as exc:
                    print('{err}. Therefore '.format(
                        err=exc) + name + " can not take this model into consideration of model selection")
        else:
            for name, model in self.models.items():
                try:
                    Y_pred = model.predict(X_val)
                    eva_dict[name] = U.regressor_evaluation(Y_val, Y_pred)
                    result = U.get_score(eva_dict[name], self.metric_type)

                    all_models_score_df[name + "_score"] = [result]

                    if "error" in self.metric_type:
                        if result < self.best_score:
                            self.best_score = result
                            best_name = name
                            best_model = model
                    else:
                        if result > self.best_score:
                            self.best_score = result
                            best_name = name
                            best_model = model


                except Exception as exc:
                    print('{err}. Therefore '.format(
                        err=exc) + name + " can not take this model into consideration of model selection")

        self.best_model[best_name] = best_model
        eva_dict = self.myEvaluation_report(X_test, Y_test, self.best_model)
        if self.models_type == "Classification":
            best_model_eva = (best_name, eva_dict[best_name])
        else:
            best_model_eva = (best_name, eva_dict)

        return all_models_score_df, best_model_eva
