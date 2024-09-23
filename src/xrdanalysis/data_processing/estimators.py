"""
Predictors for measurements, patients
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
from xrdanalysis.data_processing.containers import MLCluster
from xrdanalysis.data_processing.utility_functions import prep

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PredictorRF(BaseEstimator, ClassifierMixin):
    """
    Predictor based on RF for that is being fit to individual measurements
    """
    SCALED_DATA = 'radial_profile_data_norm_scaled'

    def __init__(self, func,
                 n_estimators=100,
                 max_depth=10,
                 random_state=32,
                 split=0.4, shuffle=True,
                 split_func=None,
                 show_result=True):
        self.func = func
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.split = split
        self.shuffle = shuffle
        self.split_func = split_func
        self.show_result = show_result

    @staticmethod
    def prep_data(df: pd.DataFrame):
        return prep(df)

    def fit(self, df: pd.DataFrame):
        dfc = df.copy()
        dfc['q_range_max'] = dfc['q_range'].apply(lambda x: np.max(x))
        dfc['q_cluster_label'] = dfc['type_measurement'].apply(lambda x: 0 if x == 'SAXS' else 1)

        dfs = dfc[dfc['type_measurement'] == 'SAXS'].copy()
        dfw = dfc[dfc['type_measurement'] == 'WAXS'].copy()

        self.ML_saxs = MLCluster(df=dfs, q_cluster=0, q_range=dfs.q_range.iloc[0])
        self.ML_waxs = MLCluster(df=dfw, q_cluster=1, q_range=dfw.q_range.iloc[0])

        self.ML_saxs.train_func = self.func
        self.ML_waxs.train_func = self.func

        # SAXS model is trained
        self.ML_saxs.model_train(n_estimators=self.n_estimators,
                                 max_depth=self.max_depth,
                                 random_state=self.random_state,
                                 split=self.split,
                                 shuffle=self.shuffle,
                                 split_func=self.split_func)
        mls = self.ML_saxs

        # WAXS model is trained
        self.ML_waxs.model_train(n_estimators=self.n_estimators,
                                 max_depth=self.max_depth,
                                 random_state=self.random_state,
                                 split=self.split,
                                 shuffle=self.shuffle,
                                 split_func=self.split_func)
        mlw = self.ML_waxs
        if self.show_result:
            print(f"""Accuracy: {mls.accuracy}, ROC: {mls.roc_auc}, q_max: {mls.df.q_range_max.mean()}, 
            q_label {mls.df.q_cluster_label.mean()}""")
            print(f"""Accuracy: {mlw.accuracy}, ROC: {mlw.roc_auc}, q_max: {mlw.df.q_range_max.mean()}, 
            q_label {mlw.df.q_cluster_label.mean()}""")

        return self

    def predict(self, df):
        dfc = df.copy()
        dfs = dfc[dfc['type_measurement'] == 'SAXS'].copy()
        dfw = dfc[dfc['type_measurement'] == 'WAXS'].copy()

        saxs_pred = self.ML_saxs.model.predict(PredictorRF.prep_data(dfs))
        waxs_pred = self.ML_waxs.model.predict(PredictorRF.prep_data(dfw))

        # Create DataFrames with the predictions, keeping the original index
        saxs_pred_df = pd.DataFrame(saxs_pred, index=dfs.index)
        waxs_pred_df = pd.DataFrame(waxs_pred, index=dfw.index)

        # Concatenate both predictions and sort by the original index
        combined_pred_df = pd.concat([saxs_pred_df, waxs_pred_df]).loc[dfc.index]
        # Return the predictions as a numpy array
        return combined_pred_df.to_numpy()

    def predict_proba(self, df):
        dfc = df.copy()
        dfs = dfc[dfc['type_measurement'] == 'SAXS'].copy()
        dfw = dfc[dfc['type_measurement'] == 'WAXS'].copy()

        saxs_pred = self.ML_saxs.model.predict_proba(PredictorRF.prep_data(dfs))
        waxs_pred = self.ML_waxs.model.predict_proba(PredictorRF.prep_data(dfw))

        # Create DataFrames with the predictions, keeping the original index
        saxs_pred_df = pd.DataFrame(saxs_pred, index=dfs.index)
        waxs_pred_df = pd.DataFrame(waxs_pred, index=dfw.index)

        # Concatenate both predictions and sort by the original index
        combined_pred_df = pd.concat([saxs_pred_df, waxs_pred_df]).loc[dfc.index]
        # Return the predictions as a numpy array
        return combined_pred_df.to_numpy()


class PredictorRFPatients(BaseEstimator, ClassifierMixin):
    """
    Predictor based on RF for that is being fit to patient
    """
    SCALED_DATA = 'radial_profile_data_norm_scaled'

    def __init__(self,
                 n_estimators_first=100,
                 n_estimators_second=100,
                 max_depth_first=10,
                 max_depth_second=10,
                 random_state=32,
                 split=0.4,
                 show_result=True):
        self.n_estimators_first = n_estimators_first
        self.n_estimators_second = n_estimators_first
        self.max_depth_first = max_depth_first
        self.max_depth_second = max_depth_second
        self.random_state = random_state
        self.split = split
        self.show_result = show_result

    def prep_data(self, df: pd.DataFrame, fit=True):
        cl_saxs, cl_waxs = self.cl_saxs, self.cl_waxs

        def flatten(mixed_list):
            flat_list = []
            for item in mixed_list:
                if isinstance(item, np.ndarray):
                    flat_list.extend(item)
                elif isinstance(item, list):
                    flat_list.extend(flatten(item))
                else:
                    flat_list.append(item)
            return flat_list

        def calc_dist(points):
            if len(points) == 0:
                return -1
            dist_matrix = distance_matrix(points, points)
            # Get the upper triangle of the distance matrix without the diagonal
            upper_triangle = dist_matrix[np.triu_indices(len(points), k=1)]
            m = np.mean(upper_triangle)
            return m

        patients_x = []
        patients_y = []

        for pat_id in df['patient_id'].unique():
            if fit:
                value = int(df[df['patient_id'] == pat_id]['cancer_diagnosis'].iloc[0])
                patients_y.append(value)

            saxs = df[(df['patient_id'] == pat_id) & (df['type_measurement'] == 'SAXS')]
            waxs = df[(df['patient_id'] == pat_id) & (df['type_measurement'] == 'WAXS')]

            try:
                age = saxs['age'].iloc[0]
            except Exception:
                age = -1

            saxs_pred = [-1, -1]
            waxs_pred = [-1, -1]
            transformed_data_saxs = []
            transformed_data_waxs = []

            if saxs.shape[0] > 0:
                transformed_data_saxs = np.vstack(saxs[PredictorRFPatients.SCALED_DATA].values)
                saxs_pred = np.mean(cl_saxs.predict_proba(transformed_data_saxs), axis=0)

            if waxs.shape[0] > 0:
                transformed_data_waxs = np.vstack(waxs[PredictorRFPatients.SCALED_DATA].values)
                waxs_pred = np.mean(cl_waxs.predict_proba(transformed_data_waxs), axis=0)

            if saxs.shape[0] > 0:
                s = 1 / saxs.shape[0]
            else:
                s = 0

            if waxs.shape[0] > 0:
                w = 1 / waxs.shape[0]
            else:
                w = 0

            res = flatten([saxs_pred,
                           waxs_pred,
                           s, w,
                           calc_dist(transformed_data_saxs),
                           calc_dist(transformed_data_waxs),
                           1 / age])

            patients_x.append(res)
        if fit:
            return np.array(patients_x), np.array(patients_y)
        else:
            return np.array(patients_x)

    def _first_layer(self, df):
        # Get unique patients and their corresponding diagnosis
        unique_patients = df[['patient_id', 'cancer_diagnosis']].drop_duplicates()

        # Perform a stratified split based on the diagnosis_encoded
        train_patients, test_patients = train_test_split(unique_patients,
                                                         test_size=self.split,
                                                         stratify=unique_patients['cancer_diagnosis'],
                                                         random_state=self.random_state
                                                         )
        # Split the original DataFrame into train and test sets based on patient IDs
        df_tr = df[df['patient_id'].isin(train_patients['patient_id'])]
        df_te = df[df['patient_id'].isin(test_patients['patient_id'])]

        rf_params = {'n_estimators': self.n_estimators_first,
                     'max_depth': self.max_depth_first,
                     'min_samples_split': 2,
                     'min_samples_leaf': 1,
                     'bootstrap': True,
                     'random_state': self.random_state
                     }
        cl_saxs = RandomForestClassifier(**rf_params)
        cl_waxs = RandomForestClassifier(**rf_params)

        df_tr_saxs = df_tr[df_tr['type_measurement'] == 'SAXS']
        df_tr_waxs = df_tr[df_tr['type_measurement'] == 'WAXS']

        y_data_saxs = df_tr_saxs["cancer_diagnosis"].astype(int).values
        y_data_waxs = df_tr_waxs["cancer_diagnosis"].astype(int).values

        transformed_data_saxs = np.vstack(df_tr_saxs[PredictorRFPatients.SCALED_DATA].values)
        transformed_data_waxs = np.vstack(df_tr_waxs[PredictorRFPatients.SCALED_DATA].values)

        cl_saxs.fit(transformed_data_saxs, y_data_saxs)
        cl_waxs.fit(transformed_data_waxs, y_data_waxs)
        self.cl_saxs = cl_saxs
        self.cl_waxs = cl_waxs
        return df_tr, df_te

    def _second_layer(self, df):

        patients_x, patients_y = self.prep_data(df)
        rf_params = {'n_estimators': self.n_estimators_second,
                     'max_depth': self.max_depth_second,
                     'min_samples_split': 2,
                     'min_samples_leaf': 1,
                     'bootstrap': True,
                     'random_state': self.random_state
                     }

        cl_decider = RandomForestClassifier(**rf_params)
        cl_decider.fit(patients_x, patients_y)

        return cl_decider

    def fit(self, df: pd.DataFrame):
        dfc = df.copy()
        df_tr, df_te = self._first_layer(dfc)
        cl_decider = self._second_layer(df_tr)

        self.cl_decider = cl_decider

        x_test, self.y_test = self.prep_data(df_te)

        y_pred = cl_decider.predict(x_test)
        y_proba = cl_decider.predict_proba(x_test)
        self.y_score = y_proba[:, 1]

        self.fpr, self.tpr, self.threshold = roc_curve(self.y_test, self.y_score)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.roc_auc = roc_auc_score(self.y_test, self.y_score)

        self.optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal_threshold = self.threshold[self.optimal_idx]
        self.optimal_sensitivity = self.tpr[self.optimal_idx]
        self.optimal_specificity = 1 - self.fpr[self.optimal_idx]

        if self.show_result:
            print(f"""Accuracy: {self.accuracy}, ROC: {self.roc_auc}""")

        return self

    def predict(self, df):
        dfc = df.copy()
        x = self.prep_data(dfc, fit=False)
        y_pred = self.cl_decider.predict(x)
        return y_pred

    def predict_proba(self, df):
        dfc = df.copy()
        x = self.prep_data(dfc, fit=False)
        y_pred_proba = self.cl_decider.predict_proba(x)
        return y_pred_proba
