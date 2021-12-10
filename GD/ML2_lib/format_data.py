import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ML2_lib import models
from ML2_lib import SGDByTorch
import torch.nn.functional as F
import torch
from sklearn.preprocessing import LabelBinarizer

class Format:
    def __init__(self):
        pass

    def


pd.read_csv("data/adult/adult_classification.csv", header=None)


def make_col_names():
    col_names = []
    continuous_dict = {}
    for i, line in enumerate(open("data/adult/adult_classification.names", "r"), 1):
        if i > 96:
            line = line.rstrip()
            name = line.split(":")[0]
            col_names.append(name)
            line = line.replace(" ", "").replace(".", "")
            continuous = line.split(":")[1] == "continuous"
            continuous_dict[name] = continuous
    col_names.append("label")
    continuous_dict["label"] = False
    return col_names, continuous_dict


n_id = 0
col_names, continuous_dicts = make_col_names()
col_names


def load_data(filename, col_names, n, skiprows=None):
    df = pd.read_csv(filename, header=None, index_col=None, skiprows=skiprows)
    # Display the number of records before delete missing valeus.
    print("the number of {} records:{}\n".format(filename, len(df.index)))
    df.columns = col_names

    # Replace the missing value's character to np.nan.
    df = df.applymap(lambda d: np.nan if d == " ?" else d)

    # Unify the different written forms.
    df = df.applymap(lambda d: " <=50K" if d == " <=50K." else d)
    df = df.applymap(lambda d: " >50K" if d == " >50K." else d)

    # Display the information about missing values and
    print("missing value info:\n{}\n".format(df.isnull().sum(axis=0)))
    df = df.dropna(axis=0)

    # the number of records after delete missing valeus.
    print("the number of {} records after trimming:{}\n".format(filename, len(df.index)))
    ids = list(np.arange(n, n + len(df.index)))
    df["ID"] = np.array(ids)
    n = n + len(df.index)
    return df, n



df_train, n_id_train = load_data("data/adult/adult_classification.csv", col_names, n_id)
df_test, n_id_test = load_data("data/adult/adult_classification.test", col_names, n_id_train, skiprows=1)



def get_not_continuous_columns(continuous_dict):
    categorical_names = [k for k, v in continuous_dict.items() if not v]
    return categorical_names



def print_labelinfo(labelnames):
    for i in range(len(labelnames)):
        print("label{}:{}".format(i, labelnames[i]))



def convert_data(df_train, df_test, n_id_train, n_id_test, continuous_dicts):
    categorical_names = get_not_continuous_columns(continuous_dicts)
    df = pd.concat((df_train, df_test), axis=0)

    # Get the dummy for the categorical data.
    for name in categorical_names:
        print(name)
        if name == "label":
            # ここではOneHotEncordingではなくLabelEncording
            # F .nll_lossの関係で
            lb = LabelBinarizer()
            dummy_df = lb.fit_transform(df[name])
            df["label"] = dummy_df
        else:
            dummy_df = pd.get_dummies(df[name])
            df = pd.concat((df, dummy_df), axis=1)
            df = df.drop(name, axis=1)

    # Convert the data type.
    for name in df.columns:
        df[name] = df[name].astype(float)

    # Reguralize the data.
    for name in df.columns:
        if name == "ID":
            df[name] = df[name]
        else:
            df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())

    df_train = df[df["ID"] < n_id_train].drop("ID", axis=1)
    df_test = df[df["ID"] >= n_id_train].drop("ID", axis=1)

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    X_train = df_train.drop("label", axis=1).values
    X_test = df_test.drop("label", axis=1).values
    return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = convert_data(df_train, df_test, n_id_train, n_id_test, continuous_dicts)
print("X_train:{} y_train:{} X_test:{} y_test:{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
