import os
import csv
import warnings
import pandas as pd
import Utilities as U
from Imputation import Imputation
from Classifiers_Regressors import Classifiers_Regressors, adapting_models
from FeatureExtraction import FeatureExtraction_Methods
from FeatureSelection import FeatureSelection_Methods

warnings.filterwarnings("ignore")


class ComputeOutput:
    def __init__(self,
                 test_size_entry_field,
                 val_size_entry_field,
                 drop_cols_entry_field,
                 Classifiers_Regressors_choice,
                 label_name_entry_field,
                 extraction_choice,
                 selection_choice,
                 adaptation_choice,
                 personalization_name_entry_field,
                 saving_choice1,
                 saving_choice2,
                 filePath_trainedModels,
                 fit_choice,
                 predict_choice1,
                 partialFit_choice1,
                 save_models_options,
                 filePath_partialFit,
                 filePath_rowData,
                 imputation_options,
                 norm_options,
                 who_first_var,
                 filePath_predictData,
                 task_type_choice,
                 method_metric_choice,
                 model_selection_choice,
                 model_metric_choice_options,
                 filePath_adapted_previous_info,
                 filePath_incremented_previous_info
                 ):
        self.test_size_entry_field = test_size_entry_field.get()
        self.val_size_entry_field = val_size_entry_field.get()
        self.drop_cols_entry_field = drop_cols_entry_field.get()
        self.Classifiers_Regressors_choice = Classifiers_Regressors_choice.get()
        self.label_name_entry_field = label_name_entry_field.get()
        self.extraction_choice = extraction_choice.get()
        self.selection_choice = selection_choice.get()
        self.adaptation_choice = adaptation_choice.get()
        self.personalization_name_entry_field = personalization_name_entry_field.get()
        self.saving_choice1 = saving_choice1.get()
        self.saving_choice2 = saving_choice2.get()
        self.filePath_trainedModels = filePath_trainedModels
        self.fit_choice = fit_choice.get()
        self.predict_choice1 = predict_choice1.get()
        self.partialFit_choice1 = partialFit_choice1.get()
        self.save_models_options = save_models_options.get()
        self.filePath_partialFit = filePath_partialFit
        self.filePath_rowData = filePath_rowData
        self.imputation_options = imputation_options.get()
        self.norm_options = norm_options.get()
        self.who_first_var = who_first_var.get()
        self.filePath_predictData = filePath_predictData
        self.task_type_choice = task_type_choice.get()
        self.method_metric_choice = method_metric_choice.get()
        self.model_selection_choice = model_selection_choice.get()
        self.model_metric_choice_options = model_metric_choice_options.get()
        self.filePath_adapted_previous_info = filePath_adapted_previous_info
        self.filePath_incremented_previous_info = filePath_incremented_previous_info
        self.run()

    def run(self):
        global avg_first_report, avg_second_report, first_small_evaluation_report_by_person, \
            second_small_evaluation_report_by_person, first_evaluation_report_by_person, \
            second_evaluation_report_by_person, myReport, org_df, selection_flag, selection_op_dict, both_flag, \
            extraction_op_dict, Y_Test, per_col_test, X_Test, extraction_flag, X_Train, Y_Train, X_Val, Y_Val

        # prepare label and personalization_col
        if not self.label_name_entry_field:
            label_col_name = None
        else:
            label_col_name = self.label_name_entry_field

        if not self.personalization_name_entry_field:
            per_col_name = None
        else:
            per_col_name = self.personalization_name_entry_field

        # preparing test size
        if not self.test_size_entry_field:
            test_size = 0.3
        else:
            assert isinstance(float(self.test_size_entry_field),
                              float) and 1 > float(
                self.test_size_entry_field) > 0, "Test size must be a float and smaller than 1 and greater than 0, default 0.3"
            test_size = float(self.test_size_entry_field)

        # preparing val size
        if not self.val_size_entry_field:
            val_size = 0.3
        else:
            assert isinstance(float(self.val_size_entry_field),
                              float) and 1 > float(
                self.val_size_entry_field) > 0, "Validation size must be a float and smaller than 1 and greater than 0, default 0.3"
            val_size = float(self.val_size_entry_field)

        # preparing columns to be dropped
        col_to_drop = self.drop_cols_entry_field.split()

        # some assertions
        if self.task_type_choice == "Not specified" and (self.Classifiers_Regressors_choice, self.model_selection_choice or self.extraction_choice or self.selection_choice or self.adaptation_choice):
            raise Exception("Task type should be specified.")

        if self.task_type_choice == "Classification" and (self.extraction_choice or self.selection_choice) and \
                self.method_metric_choice not in ["accuracy", "weighted_precision", "weighted_recall",
                                                  "weighted_f1score"]:
            raise Exception(
                "Feature engineering metric for classification must be accuracy, weighted_precision, weighted_recall or weighted_f1score")

        if self.task_type_choice == "Classification" and self.model_selection_choice and \
                self.model_metric_choice_options not in ["accuracy", "weighted_precision", "weighted_recall",
                                                         "weighted_f1score"]:
            raise Exception(
                "Model selection metric for classification must be accuracy, weighted_precision, weighted_recall or weighted_f1score")

        if self.task_type_choice == "Regression" and (self.extraction_choice or self.selection_choice) and \
                self.method_metric_choice not in ["r2_score", "median_absolute_error", "mean_squared_error",
                                                  "mean_absolute_error"]:
            raise Exception(
                "Feature engineering metric for regression must be r2_score, median_absolute_error, mean_squared_error, or mean_absolute_error")

        if self.task_type_choice == "Regression" and self.model_selection_choice and \
                self.model_metric_choice_options not in ["r2_score", "median_absolute_error", "mean_squared_error",
                                                         "mean_absolute_error"]:
            raise Exception(
                "Model selection metric for regression must be r2_score, median_absolute_error, mean_squared_error, or mean_absolute_error")

        if label_col_name is None and (
                self.adaptation_choice or self.fit_choice or self.partialFit_choice1 or self.model_selection_choice):
            raise Exception('You can not perform the selected operations on the classifiers or regressors, '
                            'because there is no label specified. '
                            'You must give label name in order to work with Classifiers or Regressors.')

        if self.filePath_adapted_previous_info and self.filePath_trainedModels:
            raise Exception(
                'You can not use both adaptation and trained_models arguments together. Either use previous adapted data or trained_models.')

        if self.extraction_choice and self.selection_choice and not label_col_name:
            raise Exception(
                'Feature Extraction and selection can not be used after each other, because there is not Label specified. Compute each one separately.')

        if self.adaptation_choice and not per_col_name:
            raise Exception('Models adaptation can not be computed without personalization column.')

        if (self.saving_choice1 or self.saving_choice2) and \
                (not self.extraction_choice and not self.selection_choice):
            raise Exception('--feature_extraction or --feature_selection or both flags require to be called first.')

        if (self.filePath_trainedModels or self.fit_choice or self.predict_choice1 or
            self.partialFit_choice1 or self.save_models_options != "Do not save" or
            self.model_selection_choice or self.adaptation_choice) and not self.Classifiers_Regressors_choice:
            raise Exception('The --Classifiers and Regressors argument requires to be called first.')

        if self.Classifiers_Regressors_choice and self.adaptation_choice and (self.fit_choice or self.predict_choice1 or
                                                                              self.partialFit_choice1 or self.model_selection_choice):
            raise Exception(
                'Other functions can not be performed with models adaptation. Models adaptation must be done alone.')

        if self.Classifiers_Regressors_choice and (self.filePath_partialFit and not self.partialFit_choice1):
            raise Exception('The --partial_fit argument requires to be also called.')

        if "results" not in os.listdir():
            os.mkdir("results")

        # read data file and name of label column
        Label_name = label_col_name

        if self.filePath_rowData:
            org_df = pd.read_csv(self.filePath_rowData).reset_index(drop=True)

            # compute imputation if it's selected
            if self.imputation_options != "Not specified":
                imp = Imputation(org_df)
                if self.imputation_options == "simple":
                    org_df = imp.SimImp()
                elif self.imputation_options == "multivariate":
                    org_df = imp.MultiImp()
                else:
                    org_df = imp.KNNImp()

            # split the data according to chosen parameters
            X_Train, Y_Train, X_Val, Y_Val, X_Test, Y_Test, per_col_train, per_col_val, per_col_test = \
                U.splitting_and_reconstruct_dataframe(
                    dataframe=org_df, Y=Label_name, val_size=val_size, test_size=test_size,
                    per_col=per_col_name, task_type=self.task_type_choice,
                    data_preprocessing=self.norm_options, drop_columns=col_to_drop)

            selection_op_dict = {}
            extraction_op_dict = {}
            selection_flag = False
            extraction_flag = False
            both_flag = False
            if self.extraction_choice and self.selection_choice:
                both_flag = True
                if self.who_first_var == "selection":
                    ins_FS = FeatureSelection_Methods(X_Train, X_Val, Y_Train=Y_Train, Y_Val=Y_Val,
                                                      save_op_csv=self.saving_choice1,
                                                      save_op_pickle=self.saving_choice2,
                                                      metric=self.method_metric_choice,
                                                      task_type=self.task_type_choice)
                    selection_op_dict = ins_FS.grid_search_FS()
                    extract_after = FeatureExtraction_Methods(
                        selection_op_dict["best_selected_features_from_training_data"],
                        selection_op_dict['best_selected_features_from_validation_data'],
                        X_Test[selection_op_dict["features_name_with_best_score"]],
                        Y_Train=Y_Train, Y_Val=Y_Val, Y_Test=Y_Test,
                        save_op_csv=self.saving_choice1,
                        save_op_pickle=self.saving_choice2, metric=self.method_metric_choice,
                        task_type=self.task_type_choice)
                    extraction_op_dict = extract_after.grid_search_FE()
                    extraction_flag = True
                else:
                    ins_FE = FeatureExtraction_Methods(X_Train, X_Val, X_Test, Y_Train=Y_Train, Y_Val=Y_Val,
                                                       Y_Test=Y_Test,
                                                       save_op_csv=self.saving_choice1,
                                                       save_op_pickle=self.saving_choice2,
                                                       metric=self.method_metric_choice,
                                                       task_type=self.task_type_choice)
                    extraction_op_dict = ins_FE.grid_search_FE()
                    select_after = FeatureSelection_Methods(
                        extraction_op_dict["extracted_training_features_with_best_score"],
                        extraction_op_dict['extracted_validation_features_with_best_score'],
                        Y_Train=Y_Train, Y_Val=Y_Val, save_op_csv=self.saving_choice1,
                        save_op_pickle=self.saving_choice2, metric=self.method_metric_choice,
                        task_type=self.task_type_choice)
                    selection_op_dict = select_after.grid_search_FS()
                    selection_flag = True
            elif self.extraction_choice and not self.selection_choice:
                ins_FE = FeatureExtraction_Methods(X_Train, X_Val, X_Test, Y_Train=Y_Train, Y_Val=Y_Val, Y_Test=Y_Test,
                                                   save_op_csv=self.saving_choice1,
                                                   save_op_pickle=self.saving_choice2,
                                                   metric=self.method_metric_choice,
                                                   task_type=self.task_type_choice)
                extraction_op_dict = ins_FE.grid_search_FE()
                extraction_flag = True
            elif self.selection_choice and not self.extraction_choice:
                ins_FS = FeatureSelection_Methods(X_Train, X_Val, Y_Train=Y_Train, Y_Val=Y_Val,
                                                  save_op_csv=self.saving_choice1,
                                                  save_op_pickle=self.saving_choice2,
                                                  metric=self.method_metric_choice,
                                                  task_type=self.task_type_choice)
                selection_op_dict = ins_FS.grid_search_FS()
                selection_flag = True

        if self.Classifiers_Regressors_choice:

            models = Classifiers_Regressors(self.task_type_choice, self.model_metric_choice_options)
            if self.filePath_rowData:
                if selection_flag:
                    X_Train = selection_op_dict["best_selected_features_from_training_data"]
                    X_Val = selection_op_dict["best_selected_features_from_validation_data"]
                    if both_flag:
                        X_Test = extraction_op_dict["extracted_test_features_with_best_score"][
                            selection_op_dict["features_name_with_best_score"]]
                        Y_Test = Y_Test.reset_index(drop=True)
                        per_col_test = per_col_test.reset_index(drop=True)
                    else:
                        X_Test = X_Test[selection_op_dict["features_name_with_best_score"]]

                elif extraction_flag:
                    X_Train = extraction_op_dict["extracted_training_features_with_best_score"]
                    X_Val = extraction_op_dict["extracted_validation_features_with_best_score"]
                    X_Test = extraction_op_dict["extracted_test_features_with_best_score"]
                    Y_Test = Y_Test.reset_index(drop=True)
                    per_col_test = per_col_test.reset_index(drop=True)

            if self.filePath_trainedModels:
                models.loading_models(self.filePath_trainedModels)

            if self.adaptation_choice:
                adapting_models(self.task_type_choice, self.model_metric_choice_options, X_Train, Y_Train, X_Test,
                                Y_Test, X_Val, Y_Val, per_col_name, per_col_test, test_size,
                                self.filePath_adapted_previous_info)

            if self.fit_choice:
                models.myFit(X_Train, Y_Train)

            if self.partialFit_choice1:
                partialFit_df = pd.read_csv(self.filePath_partialFit)
                models.end_to_end_partial_fit(partialFit_df.drop([Label_name], axis=1),
                                              partialFit_df[Label_name],
                                              X_Val, Y_Val, X_Test, Y_Test,
                                              self.filePath_incremented_previous_info)

            if self.predict_choice1:
                df_to_predict = pd.read_csv(self.filePath_predictData)
                predicted_data = models.myPredict(df_to_predict)
                with open("results/predicted_data.csv", "w") as outfile:
                    writer = csv.writer(outfile)
                    key_list = list(predicted_data.keys())
                    limit = len(key_list)
                    writer.writerow(predicted_data.keys())
                    for i in range(limit):
                        writer.writerow([predicted_data[x][i] for x in key_list])

            if self.model_selection_choice:
                all_models_score_df, myReport = models.best_model_selection(X_Val, Y_Val, X_Test, Y_Test)

                classification_report_df = pd.DataFrame(myReport[1]).transpose()
                all_models_score_df.to_csv(
                    "results/all_models_score_based_on_" + self.model_metric_choice_options + "_using_val_data.csv")
                classification_report_df.to_csv(
                    "results/evaluation_best_model_" + myReport[0] + "_using_test_data.csv")

            if self.save_models_options == "each model alone":
                models.saving_models(False)
            elif self.save_models_options == "all models together":
                models.saving_models(True)

        print("Done!")
