import warnings
import customtkinter
import os
import tkinter as tk
from PIL import Image
from tkinter import filedialog
from ComputeOutput import ComputeOutput
from CTkMessagebox import CTkMessagebox

warnings.filterwarnings("ignore")


def show_checkmark():
    # Show some positive message with the checkmark icon
    msg = CTkMessagebox(title="Done",
                        message='The task is successfully executed. Check the results of the executed operations in "./results" folder.',
                        icon="check", option_1="Another task", option_3="Close")

    response = msg.get()

    if response == "Close":
        app.destroy()
    else:
        pass


def show_error(exc):
    # Show some error message
    msg = '{err}. '.format(err=exc) + "Do you want to close the program?"
    msg = CTkMessagebox(title="Error", icon="cancel", message=msg, option_1="Stay", option_3="Yes")
    response = msg.get()

    if response == "Yes":
        app.destroy()
    else:
        pass


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # to save paths selected by the user
        self.filePath_rowData = None
        self.filePath_trainedModels = None
        self.filePath_predictData = None
        self.filePath_partialFit = None
        self.filePath_adapted_previous_info = None
        self.filePath_incremented_previous_info = None

        # framework name, which appears on the top of the interface
        self.title("An Adaptive Incremental End-to-End ML Framework with Automated Feature Engineering")
        self.geometry("1400x500")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logo_image")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "CustomTkinter_logo_single.png")),
                                                 size=(30, 30))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        # mandatory arguments section
        self.mandatory_arguments = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                           border_spacing=30,
                                                           text="Mandatory Arguments",
                                                           fg_color="transparent", text_color=("gray10", "gray90"),
                                                           hover_color=("gray70", "gray30"),
                                                           anchor="w", command=self.mandatory_arguments_event)
        self.mandatory_arguments.grid(row=0, column=0, sticky="ew")

        # data preprocessing section
        self.data_preprocessing_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                                 border_spacing=30, text="Data Preprocessing",
                                                                 fg_color="transparent",
                                                                 text_color=("gray10", "gray90"),
                                                                 hover_color=("gray70", "gray30"), anchor="w",
                                                                 command=self.data_preprocessing_button_event)
        self.data_preprocessing_button.grid(row=1, column=0, sticky="ew")

        # extraction and selection section
        self.extraction_selection_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                                   border_spacing=30, text="Extraction and Selection",
                                                                   fg_color="transparent",
                                                                   text_color=("gray10", "gray90"),
                                                                   hover_color=("gray70", "gray30"), anchor="w",
                                                                   command=self.extraction_selection_button_event)
        self.extraction_selection_button.grid(row=2, column=0, sticky="ew")

        # classifiers and regressors section
        self.classifiers_regressors_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                                     border_spacing=30,
                                                                     text="Classifiers and Regressors",
                                                                     fg_color="transparent",
                                                                     text_color=("gray10", "gray90"),
                                                                     hover_color=("gray70", "gray30"), anchor="w",
                                                                     command=self.classifiers_regressors_button_event)
        self.classifiers_regressors_button.grid(row=3, column=0, sticky="ew")

        # create general run bottom
        self.general_run_bottom = customtkinter.CTkButton(self.navigation_frame, text="Run",
                                                          command=self.general_run_event)
        self.general_run_bottom.grid(row=4, column=0, padx=20, pady=20, sticky="s")

        # framework logo
        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  AdaptoML-UX",
                                                             image=self.logo_image,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=30, weight="bold"))
        self.navigation_frame_label.grid(row=5, column=0, padx=20, pady=20)
        ############################# create mandatory_arguments frame ####################################

        self.mandatory_arguments_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # function to browse and read file path
        def mandatory_arguments_open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[('text files', '*.csv'), ('All files', '.*')])
            if file:
                self.filePath_rowData = os.path.abspath(file.name)
                self.rowData_confirmation = customtkinter.CTkLabel(self.mandatory_arguments_frame,
                                                                   text="The file is located at : " + str(
                                                                       self.filePath_rowData))
                self.rowData_confirmation.grid(row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=3)

        # file path label
        self.browser_label = customtkinter.CTkLabel(self.mandatory_arguments_frame, text="Select CSV data file")
        self.browser_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # file path browse bottom
        self.browse_button = customtkinter.CTkButton(self.mandatory_arguments_frame, text="Browse",
                                                     command=mandatory_arguments_open_file)
        self.browse_button.grid(row=0, column=1, padx=20, pady=10)

        # label name label
        self.label_name_label = customtkinter.CTkLabel(self.mandatory_arguments_frame, text="Label column name")
        self.label_name_label.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # label name entry field
        self.label_name_entry_field = customtkinter.CTkEntry(self.mandatory_arguments_frame, width=200,
                                                             height=25, placeholder_text="label column name")
        self.label_name_entry_field.grid(row=2, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

        # personalization column name label
        self.personalization_name_label = customtkinter.CTkLabel(self.mandatory_arguments_frame,
                                                                 text="Adaptation column name")
        self.personalization_name_label.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # personalization column name entry field
        self.personalization_name_entry_field = customtkinter.CTkEntry(self.mandatory_arguments_frame,
                                                                       placeholder_text="Adaptation column name")
        self.personalization_name_entry_field.grid(row=3, column=1, columnspan=2, padx=20, pady=20, sticky="ew")

        # task type label
        self.task_type_label = customtkinter.CTkLabel(self.mandatory_arguments_frame,
                                                      text="This data is for (task type)")
        self.task_type_label.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        # task type choices
        self.task_type_choice = customtkinter.CTkOptionMenu(self.mandatory_arguments_frame, variable=None,
                                                            values=["Not specified", "Classification", "Regression"])
        self.task_type_choice.grid(row=4, column=1, padx=20, pady=20, columnspan=1, sticky="ew")

        # reset inputs
        def my_reset():
            self.filePath_rowData = None
            self.label_name_entry_field.delete(0, 'end')
            self.personalization_name_entry_field.delete(0, 'end')
            self.rowData_confirmation.destroy()
            self.task_type_choice.set("Not specified")

        self.browse_button = customtkinter.CTkButton(self.mandatory_arguments_frame, text="reset inputs",
                                                     command=my_reset)
        self.browse_button.grid(row=5, column=1, padx=20, pady=10)

        ############################# create data preprocessing frame ####################################
        self.data_preprocessing_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # imputation choice label
        self.imputation_label = customtkinter.CTkLabel(self.data_preprocessing_frame, text="Imputation method")
        self.imputation_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # imputation choice combo box
        self.imputation_options = customtkinter.CTkOptionMenu(self.data_preprocessing_frame, variable=None,
                                                              values=["Not specified", "simple", "multivariate",
                                                                      "KNNImputer"])
        self.imputation_options.grid(row=0, column=1, padx=20, pady=20, columnspan=1, sticky="ew")

        # Normalization choice label
        self.Norm_label = customtkinter.CTkLabel(self.data_preprocessing_frame, text="Normalization type")
        self.Norm_label.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        # Normalization choice combo box
        self.norm_options = customtkinter.CTkOptionMenu(self.data_preprocessing_frame, variable=None,
                                                        values=["Not specified", "Normalizer", "StandardScaler",
                                                                "MinMaxScaler",
                                                                "MaxAbsScaler", "RobustScaler"])
        self.norm_options.grid(row=1, column=1, padx=20, pady=20, columnspan=1, sticky="ew")

        # test size label
        self.test_size_label = customtkinter.CTkLabel(self.data_preprocessing_frame, text="Test size")
        self.test_size_label.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        # test size entry field
        self.test_size_entry_field = customtkinter.CTkEntry(self.data_preprocessing_frame, width=400, height=25,
                                                            placeholder_text="test size must be smaller than 1, default 0.3")
        self.test_size_entry_field.grid(row=2, column=1, columnspan=3, padx=20, pady=20, sticky="ew")

        # val size label
        self.val_size_label = customtkinter.CTkLabel(self.data_preprocessing_frame, text="Validation size")
        self.val_size_label.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        # val size entry field
        self.val_size_entry_field = customtkinter.CTkEntry(self.data_preprocessing_frame,
                                                           placeholder_text="validation size must be smaller than 1, default 0.3")
        self.val_size_entry_field.grid(row=3, column=1, columnspan=3, padx=20, pady=20, sticky="ew")

        # drop cols label
        self.drop_cols_label = customtkinter.CTkLabel(self.data_preprocessing_frame,
                                                      text="Columns to be initially removed")
        self.drop_cols_label.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        # drop cols entry field
        self.drop_cols_entry_field = customtkinter.CTkEntry(self.data_preprocessing_frame,
                                                            placeholder_text="names of columns to remove, leave space between each name")
        self.drop_cols_entry_field.grid(row=4, column=1, columnspan=3, padx=20, pady=20, sticky="ew")

        # reset inputs
        def my_reset():
            self.imputation_options.set("Not specified")
            self.norm_options.set("Not specified")
            self.test_size_entry_field.delete(0, 'end')
            self.val_size_entry_field.delete(0, 'end')
            self.drop_cols_entry_field.delete(0, 'end')

        self.browse_button = customtkinter.CTkButton(self.data_preprocessing_frame, text="reset inputs",
                                                     command=my_reset)
        self.browse_button.grid(row=5, column=1, padx=20, pady=10)
        ############################# create extraction and selection ####################################

        self.extraction_selection_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # extraction activation label
        self.extraction_label = customtkinter.CTkLabel(self.extraction_selection_frame,
                                                       text="Feature extraction methods")
        self.extraction_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # extraction activation radio buttons
        self.extraction_activation_var = tk.StringVar(value=False)

        self.extraction_choice = customtkinter.CTkCheckBox(self.extraction_selection_frame,
                                                           text="Include",
                                                           variable=self.extraction_activation_var,
                                                           onvalue=True, offvalue=False)
        self.extraction_choice.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

        # selection activation label
        self.selection_label = customtkinter.CTkLabel(self.extraction_selection_frame, text="Feature selection methods")
        self.selection_label.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        # selection activation radio buttons
        self.selection_activation_var = tk.StringVar(value=False)

        self.selection_choice = customtkinter.CTkCheckBox(self.extraction_selection_frame, text="Include",
                                                          variable=self.selection_activation_var,
                                                          onvalue=True, offvalue=False)
        self.selection_choice.grid(row=1, column=1, padx=20, pady=20, sticky="ew")

        # who first radio buttons
        self.who_first_var = tk.StringVar(value="selection")

        self.selection_first_radio_button = customtkinter.CTkRadioButton(self.extraction_selection_frame,
                                                                         text="Perform feature selection first",
                                                                         variable=self.who_first_var,
                                                                         value="selection")
        self.selection_first_radio_button.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        self.extraction_first_radio_button = customtkinter.CTkRadioButton(self.extraction_selection_frame,
                                                                          text="Perform feature extraction first",
                                                                          variable=self.who_first_var,
                                                                          value="extraction")
        self.extraction_first_radio_button.grid(row=2, column=1, padx=20, pady=20, sticky="ew")

        # saving choice label
        self.saving_choice_label = customtkinter.CTkLabel(self.extraction_selection_frame,
                                                          text="Save processed feature as: ")
        self.saving_choice_label.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        # saving choice check boxes
        self.saving_choice_var = tk.StringVar(value=False)

        self.saving_choice1 = customtkinter.CTkCheckBox(self.extraction_selection_frame, text="CSV file",
                                                        variable=self.saving_choice_var, onvalue=True, offvalue=False)
        self.saving_choice1.grid(row=3, column=1, padx=20, pady=20, sticky="ew")

        self.saving_choice2 = customtkinter.CTkCheckBox(self.extraction_selection_frame, text="Pickle file",
                                                        variable=self.saving_choice_var, onvalue=True,
                                                        offvalue=False)
        self.saving_choice2.grid(row=3, column=2, padx=20, pady=20, sticky="ew")

        # metric choice label
        self.method_metric_choice_label = customtkinter.CTkLabel(self.extraction_selection_frame,
                                                                 text="Metric to select best extraction/selection method with")
        self.method_metric_choice_label.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        # metric choice options
        self.method_metric_choice_options = customtkinter.CTkOptionMenu(self.extraction_selection_frame, variable=None,
                                                                        values=["accuracy", "weighted_precision",
                                                                                "weighted_recall",
                                                                                "weighted_f1score", "r2_score",
                                                                                "median_absolute_error",
                                                                                "mean_squared_error",
                                                                                "mean_absolute_error"])
        self.method_metric_choice_options.grid(row=4, column=1, padx=20, pady=20, columnspan=1, sticky="ew")

        # reset inputs
        def my_reset():
            self.extraction_choice.deselect()
            self.selection_choice.deselect()
            self.who_first_var.set("selection")
            self.saving_choice1.deselect()
            self.saving_choice2.deselect()
            self.method_metric_choice_options.set("accuracy")

        self.browse_button = customtkinter.CTkButton(self.extraction_selection_frame, text="reset inputs",
                                                     command=my_reset)
        self.browse_button.grid(row=5, column=1, padx=20, pady=10)
        ############################# create classifiers and regressors ####################################
        self.classifiers_regressors_frame = customtkinter.CTkScrollableFrame(master=self,
                                                                             corner_radius=0, fg_color="transparent")

        # classifiers and regressors activation label
        self.classifiers_regressors_label = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                   text="Classifiers and regressors")
        self.classifiers_regressors_label.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # classifiers and regressors activation radio buttons
        self.classifiers_regressors_activation_var = tk.StringVar(value=False)

        self.classifiers_choice = customtkinter.CTkCheckBox(self.classifiers_regressors_frame, text="Include",
                                                            variable=self.classifiers_regressors_activation_var,
                                                            onvalue=True, offvalue=False)
        self.classifiers_choice.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

        # metric choice label
        self.model_metric_choice_label = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                text="Select a suitable metric to use in adaptation, incremental learning or evaluation (model selection)")
        self.model_metric_choice_label.grid(row=1, column=0, padx=20, pady=20, columnspan=2, sticky="ew")

        # metric choice options
        self.model_metric_choice_options = customtkinter.CTkOptionMenu(self.classifiers_regressors_frame, variable=None,
                                                                       values=["accuracy", "weighted_precision",
                                                                               "weighted_recall",
                                                                               "weighted_f1score", "r2_score",
                                                                               "median_absolute_error",
                                                                               "mean_squared_error",
                                                                               "mean_absolute_error"])
        self.model_metric_choice_options.grid(row=1, column=2, padx=20, pady=20, columnspan=1, sticky="ew")

        # trained models path
        def trained_models_open_file():
            self.filePath_trainedModels = filedialog.askopenfilenames(
                filetypes=[('pickle files', '*.pickle'), ('All files', '.*')])
            if self.filePath_trainedModels:
                self.trained_models_confirmation = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                          text="The file/s has/have been selected.")
                self.trained_models_confirmation.grid(row=3, column=0, padx=20, pady=20, sticky="ew", columnspan=3)

        # trained models path label
        self.trained_models_label = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                           text="Load trained models file/s")
        self.trained_models_label.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        # trained models path browse bottom
        self.trained_models_browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame, text="Browse",
                                                                    command=trained_models_open_file)
        self.trained_models_browse_button.grid(row=2, column=1, padx=20, pady=20)

        # model adaptation choice check boxes
        self.adaptation_choice_var = tk.StringVar(value=False)

        self.adaptation_choice = customtkinter.CTkCheckBox(self.classifiers_regressors_frame,
                                                           text="Perform end-to-end model adaptation",
                                                           variable=self.adaptation_choice_var,
                                                           onvalue=True, offvalue=False)
        self.adaptation_choice.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        # load previous adapted data and model
        def previous_adapted_open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[('pickle files', '*.pickle'), ('All files', '.*')])
            if file:
                self.filePath_adapted_previous_info = os.path.abspath(file.name)
                self.previous_adapted_confirmation = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                            text="The file is located at : " + str(
                                                                                self.filePath_adapted_previous_info))
                self.previous_adapted_confirmation.grid(row=6, column=0, padx=20, pady=20, sticky="ew", columnspan=3)

        # previous adapted data path label
        self.previous_adapted_label = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                             text="Load previous adapted data (.pkl) (You do no need to load saved models or train models again)")
        self.previous_adapted_label.grid(row=5, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # previous adapted data path browse bottom
        self.trained_models_browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame, text="Browse",
                                                                    command=previous_adapted_open_file)
        self.trained_models_browse_button.grid(row=5, column=2, padx=20, pady=20)

        # fit choice check boxes
        self.fit_choice_var = tk.StringVar(value=False)

        self.fit_choice = customtkinter.CTkCheckBox(self.classifiers_regressors_frame,
                                                    text="Train", variable=self.fit_choice_var,
                                                    onvalue=True, offvalue=False)
        self.fit_choice.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        # partialFit choice check boxes
        self.partialFit_choice_var = tk.StringVar(value=False)

        self.partialFit_choice1 = customtkinter.CTkCheckBox(self.classifiers_regressors_frame,
                                                            text="Batch Incremental Learning (I.e, Partial Fit)",
                                                            variable=self.partialFit_choice_var, onvalue=True,
                                                            offvalue=False)
        self.partialFit_choice1.grid(row=8, column=0, padx=20, pady=20, sticky="ew")

        def partialFit_open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[('text files', '*.csv'), ('All files', '.*')])
            if file:
                self.filePath_partialFit = os.path.abspath(file.name)
                self.partialFit_confirmation = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                      text="The File is located at : " + str(
                                                                          self.filePath_partialFit))
                self.partialFit_confirmation.grid(row=9, column=0, padx=20, pady=20, sticky="ew", columnspan=3)

        # partialFit path browse bottom
        self.partialFit_browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame,
                                                                text="Select data to partially fit models with",
                                                                command=partialFit_open_file)
        self.partialFit_browse_button.grid(row=8, column=1, padx=20, pady=20)

        # load previous incremented data and model
        def previous_increment_open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[('pickle files', '*.pickle'), ('All files', '.*')])
            if file:
                self.filePath_incremented_previous_info = os.path.abspath(file.name)
                self.previous_incremented_confirmation = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                                text="The file is located at : " + str(
                                                                                    self.filePath_incremented_previous_info))
                self.previous_incremented_confirmation.grid(row=11, column=0, padx=20, pady=20, sticky="ew",
                                                            columnspan=3)

        # previous incremented data path label
        self.previous_incremented_label = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                 text="Load previous incremented data (.pkl) (You do no need to load saved models or train models again)")
        self.previous_incremented_label.grid(row=10, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # previous incremented data path browse bottom
        self.trained_models_browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame, text="Browse",
                                                                    command=previous_increment_open_file)
        self.trained_models_browse_button.grid(row=10, column=2, padx=20, pady=20)

        # predict choice check boxes
        self.predict_choice_var = tk.StringVar(value=False)

        self.predict_choice1 = customtkinter.CTkCheckBox(self.classifiers_regressors_frame, text="Predict",
                                                         variable=self.predict_choice_var, onvalue=True, offvalue=False)
        self.predict_choice1.grid(row=12, column=0, padx=20, pady=20, sticky="ew")

        def predict_open_file():
            file = filedialog.askopenfile(mode='r', filetypes=[('text files', '*.csv'), ('All files', '.*')])
            if file:
                self.filePath_predictData = os.path.abspath(file.name)
                self.predict_confirmation = customtkinter.CTkLabel(self.classifiers_regressors_frame,
                                                                   text="The File is located at : " + str(
                                                                       self.filePath_predictData))
                self.predict_confirmation.grid(row=13, column=0, padx=20, pady=20, sticky="ew", columnspan=3)

        # predict path browse bottom
        self.trained_models_browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame,
                                                                    text="Select data to predict with",
                                                                    command=predict_open_file)
        self.trained_models_browse_button.grid(row=12, column=1, padx=20, pady=20)

        # model selection choice check boxes
        self.model_selection_choice_var = tk.StringVar(value=False)

        self.model_selection_choice = customtkinter.CTkCheckBox(self.classifiers_regressors_frame,
                                                                text="Evaluation (selecting best Classifier/Regressor based on validation data. Then evaluate that best model with test data)",
                                                                variable=self.model_selection_choice_var,
                                                                onvalue=True, offvalue=False)
        self.model_selection_choice.grid(row=14, column=0, padx=20, pady=20, columnspan=2, sticky="ew")

        # save models label
        self.save_models = customtkinter.CTkLabel(self.classifiers_regressors_frame, text="Save models")
        self.save_models.grid(row=15, column=0, padx=20, pady=20, sticky="ew")

        # save models choices
        self.save_models_options = customtkinter.CTkOptionMenu(self.classifiers_regressors_frame, variable=None,
                                                               values=["Do not save", "each model alone",
                                                                       "all models together"])
        self.save_models_options.grid(row=15, column=1, padx=20, pady=20, columnspan=2, sticky="ew")

        # reset inputs
        def my_reset():
            self.classifiers_choice.deselect()
            self.adaptation_choice.deselect()
            self.fit_choice.deselect()
            self.partialFit_choice1.deselect()
            self.predict_choice1.deselect()
            self.model_selection_choice.deselect()
            self.save_models_options.set("Not specified")
            self.model_metric_choice_options.set("accuracy")
            self.filePath_trainedModels = None
            self.filePath_partialFit = None
            self.filePath_predictData = None
            self.filePath_adapted_previous_info = None
            self.trained_models_confirmation.destroy()
            self.previous_adapted_confirmation.destroy()
            self.partialFit_confirmation.destroy()
            self.predict_confirmation.destroy()

        self.browse_button = customtkinter.CTkButton(self.classifiers_regressors_frame, text="reset inputs",
                                                     command=my_reset)
        self.browse_button.grid(row=16, column=1, padx=20, pady=10)
        ############################# select default frame ####################################

        self.select_frame_by_name("mandatory_arguments")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.mandatory_arguments.configure(
            fg_color=("gray75", "gray25") if name == "mandatory_arguments" else "transparent")
        self.data_preprocessing_button.configure(
            fg_color=("gray75", "gray25") if name == "data_preprocessing" else "transparent")
        self.extraction_selection_button.configure(
            fg_color=("gray75", "gray25") if name == "extraction_selection" else "transparent")
        self.classifiers_regressors_button.configure(
            fg_color=("gray75", "gray25") if name == "classifiers_regressors" else "transparent")

        # show selected frame
        if name == "mandatory_arguments":
            self.mandatory_arguments_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.mandatory_arguments_frame.grid_forget()
        if name == "data_preprocessing":
            self.data_preprocessing_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.data_preprocessing_frame.grid_forget()
        if name == "extraction_selection":
            self.extraction_selection_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.extraction_selection_frame.grid_forget()
        if name == "classifiers_regressors":
            self.classifiers_regressors_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.classifiers_regressors_frame.grid_forget()

    def mandatory_arguments_event(self):
        self.select_frame_by_name("mandatory_arguments")

    def data_preprocessing_button_event(self):
        self.select_frame_by_name("data_preprocessing")

    def extraction_selection_button_event(self):
        self.select_frame_by_name("extraction_selection")

    def classifiers_regressors_button_event(self):
        self.select_frame_by_name("classifiers_regressors")

    def general_run_event(self):
        try:
            ComputeOutput(self.test_size_entry_field, self.val_size_entry_field, self.drop_cols_entry_field,
                          self.classifiers_choice, self.label_name_entry_field, self.extraction_choice,
                          self.selection_choice, self.adaptation_choice, self.personalization_name_entry_field,
                          self.saving_choice1, self.saving_choice2, self.filePath_trainedModels,
                          self.fit_choice, self.predict_choice1, self.partialFit_choice1,
                          self.save_models_options, self.filePath_partialFit,
                          self.filePath_rowData, self.imputation_options, self.norm_options,
                          self.who_first_var, self.filePath_predictData,
                          self.task_type_choice, self.method_metric_choice_options, self.model_selection_choice,
                          self.model_metric_choice_options, self.filePath_adapted_previous_info,
                          self.filePath_incremented_previous_info)
            show_checkmark()
        except Exception as exc:
            show_error(exc)


if __name__ == '__main__':
    app = App()
    app.mainloop()
