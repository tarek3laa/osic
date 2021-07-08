import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch


def get_patients(train_data):
    sub = []
    for patient_id in train_data['Patient'].unique():
        sub.append(train_data[train_data['Patient'] == patient_id])
    return sub


def to_polynomial(X, degree,bias=True):
    x = []
    x.append(X.astype(np.float32))
    for i in range(1, degree):
        x.append((X ** (i + 1)).astype(np.float32))
    if bias:
      x.append(np.ones_like(X))
    x = np.vstack(x).T
    return x


def fitted_curve(patient, degree, plot=False):
    weeks = patient['Weeks']
    fvc = patient['FVC']

    X = to_polynomial(np.array(weeks), degree)

    X_inv = np.linalg.pinv(X)
    w = X_inv.dot(fvc)
    if plot:
        plt.plot(weeks, fvc, 'o')
        plt.plot(weeks, X.dot(w), 'r')
        plt.show()
    return w


def get_missing(train_data):
    missing_weeks = []
    for patient in get_patients(train_data):
        all_weeks = set(range(np.min(patient['Weeks']), np.max(patient['Weeks'])))
        missing_weeks.extend(list(all_weeks - set(patient['Weeks'])))
    return missing_weeks


def hist(data, bins, xlabel):
    plt.figure(figsize=(20, 10))
    plt.hist(data, bins, density=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.show()