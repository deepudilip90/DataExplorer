import pandas as pd
import numpy as np
from StructuredDataExplorer import StructuredDataExplorer


class PredictiveModeller(StructuredDataExplorer):

    def __init__(self, df):
        self.df = df

    def _get_cols_with_missing_numeric(self):
        self.get_fill_rate(show=False, update=True)
        missing_cols_numeric = [col for col in self.numeric_cols if self.fillrate_columns[col] < 1]

        return missing_cols_numeric

    def _get_cols_with_missing_categorical(self):

        self.get_fill_rate(show=False, update=True)

        if len(self.category_cols) < len(self.object_cols):
            self.str_to_cats()

        missing_cols_categorical = [col for col in self.category_cols if self.fillrate_columns[col] < 1]

        return missing_cols_categorical

    def _mark_rows_with_missing(self, numeric=True, categorical=False):

        if numeric:
            missing_cols_numeric = self._get_cols_with_missing_numeric()
            for col in missing_cols_numeric:
                self.df[col + '_is_na'] = np.where(pd.isnull(self.df[col]), 1, 0)
        if categorical:
            missing_cols_categorical = self._get_cols_with_missing_categorical()
            for col in missing_cols_categorical:
                self.df[col + '_is_na'] = np.where(pd.isnull(self.df[col]), 1, 0)

    def impute_median_missing_numeric(self):
        if self.fillrate_columns is None:
            self.get_fill_rate(show=False, update=True)

        missing_cols_numeric = self._get_cols_with_missing_numeric()

        if missing_cols_numeric:
            for col in missing_cols_numeric:
                self.df.loc[pd.isnull(self.df[col]), col] = self.df[col].median

    def impute_mode_missing_categorical(self):

        if self.fillrate_columns is None:
            self.get_fill_rate(show=False, update=True)

        if len(self.category_cols) < len(self.object_cols):
            self.str_to_cats()

        missing_cols_categorical = self._get_cols_with_missing_categorical()

        if missing_cols_categorical:
            modes = self.df.mode()
            for col in missing_cols_categorical:
                self.df.loc[pd.isnull(self.df[col]), col] = modes[col][0]

    def get_features_and_target(self, features=None, target=None):
        if self.target_col is None:
            if target is None:
                print('error: no taraget column specified yet!')
                return

            self.target_col = target

        if features is None:
            features = [col for col in self.df.columns if col != self.target_col]

        return self.df.loc[:, features], self.df.loc[:, target]

    def prepare_data(self):

        if len(self.category_cols) < len(self.object_cols):
            self.str_to_cats()

        self._mark_rows_with_missing()
        self.impute_median_missing_numeric()
