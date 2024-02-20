import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import datasets
from typing import List, Tuple
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split


class FeaturesData:

    def __init__(
            self,
            name: str = "iris",
            n_splits: int = 5,
            n_repeats: int = 10,
            path: Path = Path("../Datasets"),
    ):
        self.path = path
        self.name = name
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self.data_dir_path = os.path.join(self.path, self.name)
        self.data_path = os.path.join(self.data_dir_path, "data.csv")
        self.labels_path = os.path.join(self.data_dir_path, "labels.csv")
        self.data_noise_path = os.path.join(self.data_dir_path, "data_noise.csv")

        self.x = None  # features/random variables (either shuffled or not)
        self.xn = None  # noise added to the features/random variables (either shuffled or not)
        self.y = None  # targets variables/predictions (in corresponding to x)

        self.stratified_kFold_cv = None
        self.stratified_train_test_splits = defaultdict(defaultdict)

    def get_dataset(self,):

        self.x = pd.read_csv(self.data_path, header=None).values
        try:
            self.xn = pd.read_csv(self.data_noise_path, header=None).values
        except:
            self.xn = np.array([])
        self.y = pd.read_csv(self.labels_path, header=None).values.ravel()

        print("data set shape", self.x.shape, self.y.shape)

        assert not self.x.shape[0] != self.y.shape[0], "inconsistent data shapes"

        return self.x, self.xn, self.y

    def _remove_missing_data(self, df):
        for col in df.columns:
            try:
                df[col].replace({".": np.nan}, inplace=True)
                df[col].replace({"?": np.nan}, inplace=True)
            except Exception as e:
                print(e, "\n No missing values in", col)

        return df.dropna()


class FeaturesRichData:

    def __init__(
            self,
            name: str = "lawyers",
            n_splits: int = 5,
            n_repeats: int = 10,
            path: Path = Path("../Datasets"),
    ):
        self.path = path
        self.name = name
        self.n_splits = n_splits
        self.n_repeats = n_repeats

        self.data_dir_path = os.path.join(self.path, self.name)
        self.data_path_x = os.path.join(self.data_dir_path, "data_x.csv")
        self.data_path_a = os.path.join(self.data_dir_path, "data_a.csv")
        self.labels_path = os.path.join(self.data_dir_path, "labels.csv")
        self.data_noise_path_x = os.path.join(self.data_dir_path, "data_x_noise.csv")

        self.x = None  # features/random variables (either shuffled or not)
        self.a = None  # adjacency matrix (either shuffled or not)
        self.xn = None  # noise added to the features/random variables (either shuffled or not)
        self.y = None  # targets variables/predictions (in corresponding to x)

        self.stratified_kFold_cv = None
        self.stratified_train_test_splits = defaultdict(defaultdict)

    def get_dataset(self,):

        self.x = pd.read_csv(self.data_path_x, header=None).values
        self.a = pd.read_csv(self.data_path_a, header=None).values
        try:
            self.xn = pd.read_csv(self.data_noise_path_x, header=None).values
        except:
            self.xn = np.array([])
        self.y = pd.read_csv(self.labels_path, header=None).values.ravel()

        print("data set shape", self.x.shape, self.a.shape, self.y.shape)

        assert not self.x.shape[0] != self.y.shape[0], "inconsistent data shapes"

        return self.x, self.xn, self.a, self.y

    def _remove_missing_data(self, df):
        for col in df.columns:
            try:
                df[col].replace({".": np.nan}, inplace=True)
                df[col].replace({"?": np.nan}, inplace=True)
            except Exception as e:
                print(e, "\n No missing values in", col)

        return df.dropna()
