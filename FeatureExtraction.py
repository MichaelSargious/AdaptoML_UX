import pickle
from functools import reduce
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import Utilities as U


class FeatureExtraction_Methods:

    def __init__(self, X_Train,
                 X_Val,
                 X_Test,
                 Y_Train=None,
                 Y_Val=None,
                 Y_Test=None,
                 merged_columns=None,
                 method_selection=None,
                 save_op_csv=False,
                 save_op_pickle=False,
                 metric="accuracy",
                 task_type=None,

                 # Common
                 n_components=2,
                 whiten=False,
                 tol=1.0e-4,
                 random_state=None,
                 max_iter=100,
                 priors=None,
                 store_covariance=False,
                 n_jobs=None,

                 # PCA
                 copy=True,
                 svd_solver="auto",
                 iterated_power="auto",

                 # ICA
                 algorithm="parallel",
                 fun="logcosh",
                 fun_args=None,
                 w_init=None,

                 # LDA
                 solver="svd",
                 shrinkage=None,
                 covariance_estimator=None,

                 # QDA
                 reg_param=0.0,

                 # LLE
                 n_neighbors=5,
                 reg=1e-3,
                 eigen_solver="auto",
                 lle_method="standard",
                 hessian_tol=1e-4,
                 modified_tol=1e-12,
                 neighbors_algorithm="auto",

                 # t_SNE
                 perplexity=30.0,
                 early_exaggeration=12.0,
                 learning_rate=200.0,
                 n_iter=1000,
                 n_iter_without_progress=300,
                 min_grad_norm=1e-7,
                 init="random",
                 verbose=0,
                 t_sne_method="barnes_hut",
                 angle=0.5,
                 square_distances=True):

        if method_selection is None:
            method_selection = ["PCA", "ICA", "LLE"]
        if merged_columns is None:
            merged_columns = []

        # test and validation sets
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.X_Val = X_Val
        self.Y_Val = Y_Val
        self.X_Test = X_Test
        self.Y_Test = Y_Test
        self.merged_columns = merged_columns
        self.method_selection = method_selection
        self.save_op_csv = save_op_csv
        self.save_op_pickle = save_op_pickle
        self.metric = metric
        self.task_type = task_type

        # Common
        self.n_components = n_components
        self.whiten = whiten
        self.tol = tol
        self.random_state = random_state
        self.max_iter = max_iter
        self.priors = priors
        self.store_covariance = store_covariance
        self.n_jobs = n_jobs

        # PCA
        self.copy = copy
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power

        # ICA
        self.algorithm = algorithm
        self.fun = fun
        self.fun_args = fun_args
        self.w_init = w_init

        # LDA
        self.solver = solver
        self.shrinkage = shrinkage
        self.covariance_estimator = covariance_estimator

        # QDA
        self.reg_param = reg_param

        # LLE
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.lle_method = lle_method
        self.hessian_tol = hessian_tol
        self.modified_tol = modified_tol
        self.neighbors_algorithm = neighbors_algorithm

        # t_SNE
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.init = init
        self.verbose = verbose
        self.t_sne_method = t_sne_method
        self.angle = angle
        self.square_distances = square_distances

        # recombined columns
        self.rem_flag = False
        self.rem_train_df = None
        self.rem_val_df = None

        # instance for the final data frame with the desired extracted features
        self.Final_df = pd.DataFrame()

        self.extracted_training_features = {}
        self.extracted_validation_features = {}
        self.extracted_test_features = {}

        # saving information about best extracted features
        if "error" in self.metric:
            self.best_score = 10000
        else:
            self.best_score = -1
        self.name_of_best_score = ""

        self.merging_needed_columns()

    def merging_needed_columns(self):
        if self.merged_columns:
            self.rem_flag = True

            self.rem_train_df = self.X_Train[self.merged_columns]
            self.rem_train_df = self.rem_train_df.reset_index()

            self.rem_val_df = self.X_Val[self.merged_columns]
            self.rem_val_df = self.rem_val_df.reset_index()

    # implementing extraction methods customizable from the user
    def extraction_methods(self):

        # checking domains of common parameters
        assert isinstance(self.n_components, int), "n_components must be int, default = 2"
        assert isinstance(self.whiten, bool), "whiten must be bool, default = False"
        assert isinstance(self.tol, float), "tol must be float, default = 1.0e-4"

        methods_dict = {}
        methods_output_dataframes = []
        if self.rem_flag:
            methods_output_dataframes.append(self.rem_train_df)

        # specified methods
        for method in self.method_selection:

            # PCA :
            if method == "PCA" or method == "pca":
                pca = PCA(n_components=self.n_components, copy=self.copy, whiten=self.whiten,
                          svd_solver=self.svd_solver, tol=self.tol, iterated_power=self.iterated_power,
                          random_state=self.random_state)
                X_pca = pca.fit_transform(self.X_Train)

                col_names = []
                for i in range(self.n_components):
                    col_names.append("pca_n" + str(i + 1))

                df_pca = pd.DataFrame(X_pca, columns=col_names)
                df_pca = df_pca.reset_index()

                methods_dict["PCA"] = df_pca
                methods_output_dataframes.append(df_pca)

            # ICA
            elif method == "ICA" or method == "ica":
                ica = FastICA(n_components=self.n_components, algorithm=self.algorithm, fun=self.fun,
                              fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol, w_init=self.w_init,
                              random_state=self.random_state)
                X_ica = ica.fit_transform(self.X_Train)

                col_names = []
                for i in range(self.n_components):
                    col_names.append("ica_n" + str(i + 1))

                df_ica = pd.DataFrame(X_ica, columns=col_names)
                df_ica = df_ica.reset_index()

                methods_dict["ICA"] = df_ica
                methods_output_dataframes.append(df_ica)

            # LLE
            elif method == "LLE" or method == "lle":
                lle = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=self.n_components,
                                             reg=self.reg, eigen_solver=self.eigen_solver, tol=self.tol,
                                             max_iter=self.max_iter, method=self.lle_method,
                                             hessian_tol=self.hessian_tol, modified_tol=self.modified_tol,
                                             neighbors_algorithm=self.neighbors_algorithm,
                                             random_state=self.random_state, n_jobs=self.n_jobs)
                X_lle = lle.fit_transform(self.X_Train)

                col_names = []
                for i in range(self.n_components):
                    col_names.append("lle_n" + str(i + 1))

                df_lle = pd.DataFrame(X_lle, columns=col_names)
                df_lle = df_lle.reset_index()

                methods_dict["LLE"] = df_lle
                methods_output_dataframes.append(df_lle)

            # t-SNE
            """"
            elif method == "TSNE" or method == "tsne":
                tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity,
                            early_exaggeration=self.early_exaggeration, learning_rate=self.learning_rate,
                            n_iter=self.n_iter, n_iter_without_progress=self.n_iter_without_progress,
                            min_grad_norm=self.min_grad_norm,
                            init=self.init, verbose=self.verbose, random_state=self.random_state,
                            method=self.t_sne_method, angle=self.angle, n_jobs=self.n_jobs,
                            square_distances=self.square_distances)
                X_tsne = tsne.fit_transform(self.X_Train)

                col_names = []
                for i in range(self.n_components):
                    col_names.append("tsne_n" + str(i + 1))

                df_tsne = pd.DataFrame(X_tsne, columns=col_names)
                df_tsne = df_tsne.reset_index()

                methods_dict["TSNE"] = df_tsne
                methods_output_dataframes.append(df_tsne)
            
            if self.Y_Train is not None:
                if method == "LDA" or method == "lda":
                    lda = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage, priors=self.priors,
                                                     n_components=self.n_components,
                                                     store_covariance=self.store_covariance, tol=self.tol,
                                                     covariance_estimator=self.covariance_estimator)
                    X_lda = lda.fit_transform(self.X_Train, self.Y_Train)

                    col_names = []
                    for i in range(self.n_components):
                        col_names.append("lda_n" + str(i + 1))

                    df_lda = pd.DataFrame(X_lda, columns=col_names)
                    df_lda = df_lda.reset_index()

                    methods_dict["LDA"] = df_lda
                    methods_output_dataframes.append(df_lda)
             """

        if self.Y_Train is None:
            print('Warning : LDA method can not be used because there is not Y specified.')

        final_results_df = reduce(lambda left, right: pd.merge(left, right, on=['index'], how='outer'),
                                  methods_output_dataframes)
        final_results_df = final_results_df.drop(["index"], axis=1)

        if self.rem_flag:
            final_results_df = final_results_df.reset_index()
            final_results_df = reduce(lambda left, right: pd.merge(left, right, on=['index'], how='outer'),
                                      [self.rem_train_df, final_results_df])
            final_results_df = final_results_df.drop(["index"], axis=1)

        if self.save_op_csv:
            final_results_df.to_csv("results/all_extracted_training_features_in_one.csv", sep=";")

        final_dict_with_all_information = {
            "original_training_features": self.X_Train,
            "original_training_target": self.Y_Train,
            "extracted_training_features": methods_dict,
            "extracted_features_columns_names": list(methods_dict.keys())}

        if self.save_op_pickle:
            with open('results/final_dict_with_all_training_information.pickle', 'wb') as fp:
                pickle.dump(final_dict_with_all_information, fp)
            fp.close()

        return final_dict_with_all_information

    def build_all_in_one(self, extracted_val_X, extracted_train_X, extracted_test_X, score, verbose, fixed_name,
                         ncom_value):
        eva_dict = U.forest_test(extracted_val_X, self.Y_Val, self.task_type)  # getting the evaluation dict from RF
        result = U.get_score(eva_dict, score)  # get the exact score
        if verbose:
            self.Final_df[fixed_name + "_score"] = [result]
            if verbose == 1:
                comp_col_names = []
                for i in range(ncom_value):
                    comp_col_names.append(fixed_name + "_n" + str(i + 1))

                val_temp_df = pd.DataFrame(extracted_val_X, columns=comp_col_names)
                train_temp_df = pd.DataFrame(extracted_train_X, columns=comp_col_names)
                test_temp_df = pd.DataFrame(extracted_test_X, columns=comp_col_names)

                self.extracted_validation_features[fixed_name] = val_temp_df
                self.extracted_training_features[fixed_name] = train_temp_df
                self.extracted_test_features[fixed_name] = test_temp_df

                if self.rem_flag:
                    val_temp_df = val_temp_df.reset_index()
                    mdf = reduce(lambda left, right: pd.merge(left, right, on=['index'], how='outer'),
                                 [self.rem_df, val_temp_df])
                    mdf = mdf.drop(["index"], axis=1)
                    self.Final_df[fixed_name] = [mdf.values.tolist()]
                else:
                    self.Final_df[fixed_name] = [val_temp_df.values.tolist()]
        if self.task_type == "Classification":
            if self.best_score < result:
                self.best_score = result
                self.name_of_best_score = fixed_name
        else:
            if "error" in self.metric:
                if self.best_score > result:
                    self.best_score = result
                    self.name_of_best_score = fixed_name
            else:
                if self.best_score < result:
                    self.best_score = result
                    self.name_of_best_score = fixed_name

    def grid_search_FE(self, param_grid=None, verbose=1):

        global ncom_value
        if self.Y_Val is None:
            print(
                "grid_search can not be used because there is no label (Y) specified. Normal selection methods will be performed without grid search.")
            return self.extraction_methods()

        if len(self.X_Train.columns) <= 0:
            print(
                "Feature Extraction Gridsearch can not be done, because there are less than two columns in the dataset.")

            if self.save_op_csv:
                self.X_Train.to_csv("results/all_extracted_validation_features_from_gridsearch_in_one.csv", sep=";")

            final_dict_with_all_information = {
                "original_training_features": self.X_Train,
                "original_training_target": self.Y_Train,
                "original_validation_features": self.X_Val,
                "original_validation_target": self.Y_Val,
                "name_of_best_score": None,
                "extracted_training_features_with_best_score": self.X_Train,
                "extracted_validation_features_with_best_score": self.X_Val,
                "extracted_test_features_with_best_score": self.X_Test,
                "all_extracted_training_features": self.X_Train,
                "all_extracted_validation_features": self.X_Val,
                "all_extracted_test_features": self.X_Test
            }

            if self.save_op_pickle:
                with open('results/final_dict_with_all_validation_information.pickle', 'wb') as fp:
                    pickle.dump(final_dict_with_all_information, fp)
                fp.close()

            return final_dict_with_all_information

        assert verbose in [1, 2], "Verbose must be 1 or 2. Where with verbose=1, you get csv file with all features" \
                                  " and scores. With verbose=2 you get csv file with only scores of the grid search."

        if param_grid is None:
            param_grid = {
                "n_components": [],
                "max_iter": [500, 1000]
            }
        if self.X_Train.shape[0] == 1 or self.X_Train.shape[1] == 1:
            param_grid["n_components"] = [1]
        elif self.X_Train.shape[0] == 2 or self.X_Train.shape[1] == 2:
            param_grid["n_components"] = [1, 2]
        else:
            param_grid["n_components"] = U.fill_mylist(param_grid["n_components"],
                                                       min(self.X_Train.shape[0], self.X_Train.shape[1]), 2)

        if len(param_grid.keys()) == 3:
            for ncom_value in param_grid["n_components"]:
                for tol_count, tol_value in enumerate(param_grid["tol"]):
                    fixed_name = "pca_comp_" + str(ncom_value) + "_tol_" + str(tol_value)
                    pca = PCA(n_components=ncom_value, tol=tol_value)
                    pca.fit(self.X_Train)
                    extracted_val_X_pca = pca.transform(self.X_Val)
                    extracted_train_X_pca = pca.transform(self.X_Train)
                    extracted_test_X_pca = pca.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca, self.metric,
                                          verbose, fixed_name, ncom_value)

                    for miter_value in param_grid["max_iter"]:
                        fixed_name = "ica_comp_" + str(ncom_value) + "_tol_" + str(tol_value) + "_iter_" + str(
                            miter_value)
                        ica = FastICA(n_components=ncom_value, tol=tol_value, max_iter=miter_value)
                        ica.fit(self.X_Train)
                        extracted_val_X_ica = ica.transform(self.X_Val)
                        extracted_train_X_ica = ica.transform(self.X_Train)
                        extracted_test_X_ica = ica.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

                        fixed_name = "lle_comp_" + str(ncom_value) + "_tol_" + str(tol_value) + "_iter_" + str(
                            miter_value)
                        embedding = LocallyLinearEmbedding(n_components=ncom_value, eigen_solver="dense", tol=tol_value,
                                                           max_iter=miter_value)
                        embedding.fit(self.X_Train)
                        extracted_val_X_lle = embedding.transform(self.X_Val)
                        extracted_train_X_lle = embedding.transform(self.X_Train)
                        extracted_test_X_lle = embedding.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

        if len(param_grid.keys()) == 2:
            if "n_components" in param_grid.keys() and "tol" in param_grid.keys():
                for ncom_value in param_grid["n_components"]:
                    for tol_value in param_grid["tol"]:
                        fixed_name = "pca_comp_" + str(ncom_value) + "_tol_" + str(tol_value)
                        pca = PCA(n_components=ncom_value, tol=tol_value)
                        pca.fit(self.X_Train)
                        extracted_val_X_pca = pca.transform(self.X_Val)
                        extracted_train_X_pca = pca.transform(self.X_Train)
                        extracted_test_X_pca = pca.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

                        fixed_name = "ica_comp_" + str(ncom_value) + "_tol_" + str(tol_value)
                        ica = FastICA(n_components=ncom_value, tol=tol_value)
                        ica.fit(self.X_Train)
                        extracted_val_X_ica = ica.transform(self.X_Val)
                        extracted_train_X_ica = ica.transform(self.X_Train)
                        extracted_test_X_ica = ica.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

                        fixed_name = "lle_comp_" + str(ncom_value) + "_tol_" + str(tol_value)
                        embedding = LocallyLinearEmbedding(n_components=ncom_value, eigen_solver="dense", tol=tol_value)
                        embedding.fit(self.X_Train)
                        extracted_val_X_lle = embedding.transform(self.X_Val)
                        extracted_train_X_lle = embedding.transform(self.X_Train)
                        extracted_test_X_lle = embedding.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

            if "n_components" in param_grid.keys() and "max_iter" in param_grid.keys():
                for ncom_value in param_grid["n_components"]:
                    fixed_name = "pca_comp_" + str(ncom_value) + "_tol_" + "default_0.0"
                    pca = PCA(n_components=ncom_value)
                    pca.fit(self.X_Train)
                    extracted_val_X_pca = pca.transform(self.X_Val)
                    extracted_train_X_pca = pca.transform(self.X_Train)
                    extracted_test_X_pca = pca.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca, self.metric,
                                          verbose, fixed_name, ncom_value)

                    for miter_value in param_grid["max_iter"]:
                        fixed_name = "ica_comp_" + str(ncom_value) + "_iter_" + str(miter_value)
                        ica = FastICA(n_components=ncom_value, max_iter=miter_value)
                        ica.fit(self.X_Train)
                        extracted_val_X_ica = ica.transform(self.X_Val)
                        extracted_train_X_ica = ica.transform(self.X_Train)
                        extracted_test_X_ica = ica.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

                        fixed_name = "lle_comp_" + str(ncom_value) + "_iter_" + str(miter_value)
                        embedding = LocallyLinearEmbedding(n_components=ncom_value, eigen_solver="dense",
                                                           max_iter=miter_value)
                        embedding.fit(self.X_Train)
                        extracted_val_X_lle = embedding.transform(self.X_Val)
                        extracted_train_X_lle = embedding.transform(self.X_Train)
                        extracted_test_X_lle = embedding.transform(self.X_Test)
                        self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle,
                                              self.metric,
                                              verbose, fixed_name, ncom_value)

        if len(param_grid.keys()) == 1:
            if "n_components" in param_grid.keys():
                for ncom_value in param_grid["n_components"]:
                    fixed_name = "pca_comp_" + str(ncom_value)
                    pca = PCA(n_components=ncom_value)
                    pca.fit(self.X_Train)
                    extracted_val_X_pca = pca.transform(self.X_Val)
                    extracted_train_X_pca = pca.transform(self.X_Train)
                    extracted_test_X_pca = pca.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca, self.metric,
                                          verbose, fixed_name, ncom_value)

                    fixed_name = "ica_comp_" + str(ncom_value)
                    ica = FastICA(n_components=ncom_value)
                    ica.fit(self.X_Train)
                    extracted_val_X_ica = ica.transform(self.X_Val)
                    extracted_train_X_ica = ica.transform(self.X_Train)
                    extracted_test_X_ica = ica.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica, self.metric,
                                          verbose, fixed_name, ncom_value)

                    fixed_name = "lle_comp_" + str(ncom_value)
                    embedding = LocallyLinearEmbedding(n_components=ncom_value, eigen_solver="dense")
                    embedding.fit(self.X_Train)
                    extracted_val_X_lle = embedding.transform(self.X_Val)
                    extracted_train_X_lle = embedding.transform(self.X_Train)
                    extracted_test_X_lle = embedding.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle, self.metric,
                                          verbose, fixed_name, ncom_value)

            if "tol" in param_grid.keys():
                for tol_value in param_grid["tol"]:
                    fixed_name = "pca_comp_3_tol_" + str(tol_value)
                    pca = PCA(n_components=3, tol=tol_value)
                    pca.fit(self.X_Train)
                    extracted_val_X_pca = pca.transform(self.X_Val)
                    extracted_train_X_pca = pca.transform(self.X_Train)
                    extracted_test_X_pca = pca.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca, self.metric,
                                          verbose, fixed_name, ncom_value)

                    fixed_name = "ica_comp_3_tol_" + str(tol_value)
                    ica = FastICA(n_components=3, tol=tol_value)
                    ica.fit(self.X_Train)
                    extracted_val_X_ica = ica.transform(self.X_Val)
                    extracted_train_X_ica = ica.transform(self.X_Train)
                    extracted_test_X_ica = ica.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica, self.metric,
                                          verbose, fixed_name, ncom_value)

                    fixed_name = "lle_comp_3_tol_" + str(tol_value)
                    embedding = LocallyLinearEmbedding(n_components=3, tol=tol_value, eigen_solver="dense")
                    embedding.fit(self.X_Train)
                    extracted_val_X_lle = embedding.transform(self.X_Val)
                    extracted_train_X_lle = embedding.transform(self.X_Train)
                    extracted_test_X_lle = embedding.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle, self.metric,
                                          verbose, fixed_name, ncom_value)

            if "max_iter" in param_grid.keys():
                fixed_name = "pca_comp_default_3_tol_default_0.0"
                pca = PCA(n_components=3)
                pca.fit(self.X_Train)
                extracted_val_X_pca = pca.transform(self.X_Val)
                extracted_train_X_pca = pca.transform(self.X_Train)
                extracted_test_X_pca = pca.transform(self.X_Test)
                self.build_all_in_one(extracted_val_X_pca, extracted_train_X_pca, extracted_test_X_pca, self.metric,
                                      verbose, fixed_name, ncom_value)

                for miter_value in param_grid["max_iter"]:
                    fixed_name = "ica_comp_3_maxIter_" + str(miter_value)
                    ica = FastICA(n_components=3, max_iter=miter_value)
                    ica.fit(self.X_Train)
                    extracted_val_X_ica = ica.transform(self.X_Val)
                    extracted_train_X_ica = ica.transform(self.X_Train)
                    extracted_test_X_ica = ica.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_ica, extracted_train_X_ica, extracted_test_X_ica, self.metric,
                                          verbose, fixed_name, ncom_value)

                    fixed_name = "lle_comp_3_maxIter_" + str(miter_value)
                    embedding = LocallyLinearEmbedding(n_components=3, max_iter=miter_value, eigen_solver="dense")
                    embedding.fit(self.X_Train)
                    extracted_val_X_lle = embedding.transform(self.X_Val)
                    extracted_train_X_lle = embedding.transform(self.X_Train)
                    extracted_test_X_lle = embedding.transform(self.X_Test)
                    self.build_all_in_one(extracted_val_X_lle, extracted_train_X_lle, extracted_test_X_lle, self.metric,
                                          verbose, fixed_name, ncom_value)

        if self.save_op_csv:
            self.extracted_training_features[self.name_of_best_score].to_csv(
                "results/best_extracted_training_features_are_" + self.name_of_best_score + "_with_score_" + str(
                    self.best_score) + ".csv", sep=";")
            self.extracted_validation_features[self.name_of_best_score].to_csv(
                "results/best_extracted_validation_features_are_" + self.name_of_best_score + "_with_score_" + str(
                    self.best_score) + ".csv", sep=";")
            self.extracted_test_features[self.name_of_best_score].to_csv(
                "results/best_extracted_test_features_are_" + self.name_of_best_score + "_with_score_" + str(
                    self.best_score) + ".csv", sep=";")
            self.Final_df.to_csv("results/all_extracted_validation_features_from_gridsearch_in_one.csv", sep=";")

        final_dict_with_all_information = {
            "original_training_features": self.X_Train,
            "original_training_target": self.Y_Train,
            "original_validation_features": self.X_Val,
            "original_validation_target": self.Y_Val,
            "name_of_best_score": self.name_of_best_score,
            "extracted_training_features_with_best_score": self.extracted_training_features[self.name_of_best_score],
            "extracted_validation_features_with_best_score": self.extracted_validation_features[
                self.name_of_best_score],
            "extracted_test_features_with_best_score": self.extracted_test_features[self.name_of_best_score],
            "all_extracted_training_features": self.extracted_training_features,
            "all_extracted_validation_features": self.extracted_validation_features,
            "all_extracted_test_features": self.extracted_test_features
        }

        if self.save_op_pickle:
            with open('results/final_dict_with_all_validation_information.pickle', 'wb') as fp:
                pickle.dump(final_dict_with_all_information, fp)
            fp.close()

        return final_dict_with_all_information
