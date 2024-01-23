from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import random


def getting_classes_name(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def fill_mylist(myList, n, step=2):
    if n > 10:
        step = 3
    elif n == 1:
        step = 1
    myList.extend(range(1, n, step))
    return myList


# random forest classifier for evaluation.
def forest_test(X, Y, task_type):
    global report
    X_Train, X_Test, y_train, y_Test = train_test_split(X, Y, test_size=0.30, random_state=101)
    if task_type == "Regression":
        trained_forest = RandomForestRegressor(max_depth=2, random_state=0).fit(X_Train, y_train)
        prediction_forest = trained_forest.predict(X_Test)
        report = regressor_evaluation(y_Test, prediction_forest)
    elif task_type == "Classification":
        trained_forest = RandomForestClassifier(n_estimators=700).fit(X_Train, y_train)
        prediction_forest = trained_forest.predict(X_Test)
        report = classification_report(y_Test, prediction_forest, output_dict=True)
    return report


# switcher between different evaluation scores.
def get_score(dict, score):
    global result
    if score == "accuracy":
        result = dict[score]
    elif score == "macro_precision":
        result = dict["macro avg"]["precision"]
    elif score == "macro_recall":
        result = dict["macro avg"]["recall"]
    elif score == "macro_f1score":
        result = dict["macro avg"]["f1-score"]
    elif score == "weighted_precision":
        result = dict["weighted avg"]["precision"]
    elif score == "weighted_recall":
        result = dict["weighted avg"]["recall"]
    elif score == "weighted_f1score":
        result = dict["weighted avg"]["f1-score"]
    elif score == "r2_score":
        result = dict["r2_score"]
    elif score == "median_absolute_error":
        result = dict["median_absolute_error"]
    elif score == "mean_squared_error":
        result = dict["mean_squared_error"]
    elif score == "mean_absolute_error":
        result = dict["mean_absolute_error"]
    return result


def regressor_evaluation(y_true, y_pred):
    eval_dict = {
        "mean_absolute_error": round(mean_absolute_error(y_true, y_pred), 3),
        "mean_squared_error": round(mean_squared_error(y_true, y_pred), 3),
        "median_absolute_error": round(median_absolute_error(y_true, y_pred), 3),
        "r2_score": round(r2_score(y_true, y_pred), 3),
    }
    return eval_dict


# splitting data
def splitting_and_reconstruct_dataframe(dataframe,
                                        Y,
                                        val_size,
                                        test_size,
                                        task_type,
                                        per_col=None,
                                        data_preprocessing=None,
                                        drop_columns=None):
    per_col_train = None
    per_col_val = None
    per_col_test = None
    Y_Train = None
    Y_Val = None
    Y_Test = None

    if Y is None and data_preprocessing != "Not specified":
        raise Exception("Can not use data_preprocessing because there is no Y specified.")

    for col in dataframe.columns:
        if dataframe[col].dtype == "object" or dataframe[col].dtype == "bool" or dataframe[col].dtype == "category":
            dataframe[col] = pd.factorize(dataframe[col])[0]
            # dataframe = dataframe.drop([col], axis=1)

    if per_col and per_col in dataframe.columns:
        groups = dataframe.groupby(per_col)

        df_list = []
        for g in groups:
            df_list.append(g[1])

        random.Random(101).shuffle(df_list)
        test_ratio = round(test_size * len(df_list))
        df_test = pd.concat(df_list[:test_ratio])
        df_train_val = pd.concat(df_list[test_ratio:])

        if Y is not None:
            Y_Test = df_test[Y]
            X_Test = df_test.drop([Y], axis=1)

            y_train_val = df_train_val[Y]
            X_train_val = df_train_val.drop([Y], axis=1)

            if task_type == "Regression":
                X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                                  random_state=42)
            else:
                X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                                  stratify=y_train_val, random_state=42)

            if Y_Train.dtype == "object" or Y_Train.dtype == "bool" or Y_Train.dtype == "category":
                Y_Train = pd.Series(pd.factorize(Y_Train)[0])
            if Y_Val.dtype == "object" or Y_Val.dtype == "bool" or Y_Val.dtype == "category":
                Y_Val = pd.Series(pd.factorize(Y_Val)[0])
            if Y_Test.dtype == "object" or Y_Test.dtype == "bool" or Y_Test.dtype == "category":
                Y_Test = pd.Series(pd.factorize(Y_Test)[0])

        else:
            val_ratio = round(val_size * len(df_train_val))
            X_Val = df_train_val[:val_ratio]
            X_Train = df_train_val[val_ratio:]
            X_Test = df_test
    else:
        if Y is not None:
            label = dataframe[Y]
            dataframe = dataframe.drop([Y], axis=1)

            if task_type == "Regression":
                X_Train_Val, X_Test, Y_Train_Val, Y_Test = train_test_split(dataframe, label, test_size=test_size,
                                                                            random_state=42)
                X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train_Val, Y_Train_Val, test_size=val_size,
                                                                  random_state=42)
            else:
                X_Train_Val, X_Test, Y_Train_Val, Y_Test = train_test_split(dataframe, label, test_size=test_size,
                                                                            stratify=label, random_state=42)
                X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train_Val, Y_Train_Val, test_size=val_size,
                                                                  stratify=Y_Train_Val, random_state=42)

        else:
            X_Train_Val, X_Test = train_test_split(dataframe, test_size=test_size, random_state=42)
            X_Train, X_Val = train_test_split(X_Train_Val, test_size=val_size, random_state=42)

    if per_col and per_col in dataframe.columns:
        per_col_test = X_Test[per_col]
        X_Test = X_Test.drop([per_col], axis=1)
        per_col_train = X_Train[per_col]
        X_Train = X_Train.drop([per_col], axis=1)
        per_col_val = X_Val[per_col]
        X_Val = X_Val.drop([per_col], axis=1)

    if data_preprocessing != "Not specified":
        X_Train, X_Val, X_Test = preprocessing(data_preprocessing, X_Train, Y_Train, X_Val, X_Test)

    if drop_columns:
        X_Train = X_Train.drop(drop_columns, axis=1)
        X_Val = X_Val.drop(drop_columns, axis=1)
        X_Test = X_Test.drop(drop_columns, axis=1)

    return X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test, per_col_train, per_col_val, per_col_test


def preprocessing(data_preprocessing, X_Train, Y_Train, X_Val, X_Test):
    global prepro
    col_names = X_Train.columns
    if data_preprocessing == "Normalizer":
        prepro = Normalizer().fit(X_Train, Y_Train)
    elif data_preprocessing == "StandardScaler":
        prepro = StandardScaler().fit(X_Train, Y_Train)
    elif data_preprocessing == "MinMaxScaler":
        prepro = MinMaxScaler().fit(X_Train, Y_Train)
    elif data_preprocessing == "MaxAbsScaler":
        prepro = MaxAbsScaler().fit(X_Train, Y_Train)
    elif data_preprocessing == "RobustScaler":
        prepro = RobustScaler().fit(X_Train, Y_Train)

    norm_X_Train = prepro.transform(X_Train)
    X_Train = pd.DataFrame(norm_X_Train, columns=col_names)
    norm_X_Val = prepro.transform(X_Val)
    X_Val = pd.DataFrame(norm_X_Val, columns=col_names)
    norm_X_Test = prepro.transform(X_Test)
    X_Test = pd.DataFrame(norm_X_Test, columns=col_names)

    return X_Train, X_Val, X_Test


"""""
def cls_reports_average(reports_list):
    mean_dict = dict()
    for label in reports_list[0].keys():
        dictionary = dict()

        if label in 'accuracy':
            mean_dict[label] = sum(d[label] for d in reports_list) / len(reports_list)
            continue

        for key in reports_list[0][label].keys():
            dictionary[key] = sum(d[label][key] for d in reports_list if label in d.keys()) / len(reports_list)
        mean_dict[label] = dictionary

    return mean_dict
"""""


def reports_average(reports_list):
    sums = Counter()
    counters = Counter()
    for d in reports_list:
        sums.update(d)
        counters.update(d.keys())

    mean_dict = {x: round(float(sums[x]) / counters[x], 3) for x in sums.keys()}

    return mean_dict


def print_csv_files(run_dict, others_dict):
    avg_run = reports_average(list(run_dict.values())[1:])
    avg_others = reports_average(list(others_dict.values())[1:])

    run_columns_names = list(run_dict["UBM"].keys())
    others_columns_names = list(run_dict["UBM"].keys())
    run_columns_names.insert(0, "adapted_entities")
    others_columns_names.insert(0, "non_adapted_entities")

    run_df = pd.DataFrame(columns=run_columns_names)
    other_df = pd.DataFrame(columns=others_columns_names)

    run_index_name = []
    others_index_name = []
    for (run_name, run_eva), (others_name, others_eva) in zip(run_dict.items(), others_dict.items()):
        run_df = run_df.append(run_eva, ignore_index=True)
        other_df = other_df.append(others_eva, ignore_index=True)
        run_index_name.append(run_name)
        others_index_name.append(others_name)

    run_df = run_df.append(avg_run, ignore_index=True)
    other_df = other_df.append(avg_others, ignore_index=True)

    run_index_name.append("avg_w/o_UBM")
    others_index_name.append("avg_w/o_UBM")

    run_df["adapted_entities"] = run_index_name
    other_df["non_adapted_entities"] = others_index_name

    return run_df, other_df


def print_csv_increment(eva_dict):
    avg_run = reports_average(list(eva_dict.values())[1:])
    run_columns_names = list(eva_dict["base_model"].keys())
    run_columns_names.insert(0, "session_nr")

    run_df = pd.DataFrame(columns=run_columns_names)

    run_index_name = []
    for run_name, run_eva in eva_dict.items():
        run_df = run_df.append(run_eva, ignore_index=True)
        run_index_name.append(run_name)

    run_df = run_df.append(avg_run, ignore_index=True)

    run_index_name.append("avg")
    run_df["session_nr"] = run_index_name

    return run_df


def plot_all_metrics(run, others, task_type, session_nr):
    x_index = list(run.keys())

    run_by_metric = {}
    others_by_metric = {}

    for metric, value in run['UBM'].items():
        run_by_metric[metric] = []
        others_by_metric[metric] = []

    for (run_name, run_dict), (other_name, other_dict) in zip(run.items(), others.items()):
        for metric, value in run_dict.items():
            run_by_metric[metric].append(value)
        for metric, value in other_dict.items():
            others_by_metric[metric].append(value)

    if task_type == "Classification":
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
        figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), constrained_layout=True)
    else:
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1)]
        figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)

    figure.suptitle("Entities evaluation with different metrics in session " + str(session_nr))

    for (r, c), (metric, values) in zip(rows_columns_gragh, run_by_metric.items()):
        axes[r, c].plot(x_index, values, 'o-', label="adapted entities")
        axes[r, c].plot(x_index, others_by_metric[metric], 'o-', label="non adapted entities")
        axes[r, c].set_title(metric)
        axes[r, c].set_xticklabels(x_index, rotation=45, ha='right')
        axes[r, c].legend()

    plt.savefig("results/entities_and_others_evaluation_metrics_in_session_" + str(session_nr) + ".png")


def plot_sessions_graphs(persons_avg_dict, others_avg_dict, task_type):
    x_index = list(persons_avg_dict.keys())

    persons_avg_by_metric = {}
    others_avg_by_metric = {}

    for metric, value in persons_avg_dict['UBM'].items():
        persons_avg_by_metric[metric] = []
        others_avg_by_metric[metric] = []

    for (person_session_nr, persons_avg_dict), (other_session_nr, others_avg_dict) in zip(
            persons_avg_dict.items(), others_avg_dict.items()):
        for metric, value in persons_avg_dict.items():
            persons_avg_by_metric[metric].append(value)
        for metric, value in others_avg_dict.items():
            others_avg_by_metric[metric].append(value)

    if task_type == "Classification":
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 13), constrained_layout=True)
    else:
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
        figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 13), constrained_layout=True)

    figure.suptitle("Avg evaluation with different metrics in all sessions")

    for (r, c), (metric, values) in zip(rows_columns_gragh, persons_avg_by_metric.items()):
        axes[r, c].plot(x_index, values, 'o-', label="adapted entities")
        axes[r, c].plot(x_index, others_avg_by_metric[metric], 'o-', label="non adapted entities")
        axes[r, c].set_title(metric)
        axes[r, c].set_xticklabels(x_index, rotation=45, ha='right')
        axes[r, c].legend()

    plt.savefig("results/persons_and_others_avg_evaluation_all_sessions.png")


def compute_kemker_metric(eva_dict, session_nr, metric_type):

    key_list = list(eva_dict)
    temp = 0
    for current_session, current_eva in eva_dict.items():
        if current_session == "base_model":
            continue
        try:
            prev_eva = eva_dict[key_list[key_list.index(current_session) - 1]]
            temp += current_eva[metric_type]/prev_eva[metric_type]
        except (ValueError, IndexError):
            break
    eva_dict["session_" + str(session_nr)]["kemker_metric"] = temp/session_nr

    return eva_dict


def plot_incremental_graphs(eva_dict, task_type):
    x_index = list(eva_dict.keys())

    eva_by_metric = {}

    for metric, value in eva_dict['base_model'].items():
        eva_by_metric[metric] = []

    for session_nr, session_eva_dict in eva_dict.items():
        for metric, value in session_eva_dict.items():
            eva_by_metric[metric].append(value)

    if task_type == "Classification":
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 13), constrained_layout=True)
    else:
        rows_columns_gragh = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
        figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 13), constrained_layout=True)

    figure.suptitle("Avg incremental evaluation with different metrics from all sessions")

    for (r, c), (metric, values) in zip(rows_columns_gragh, eva_by_metric.items()):
        axes[r, c].plot(x_index, values, 'o-', label="avg incremental evaluation")
        axes[r, c].set_title(metric)
        axes[r, c].set_xticklabels(x_index, rotation=45, ha='right')
        axes[r, c].legend()

    plt.savefig("results/avg_incremental_evaluation_all_sessions.png")







