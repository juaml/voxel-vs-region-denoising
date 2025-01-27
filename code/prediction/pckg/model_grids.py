#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:29:12 2021
@author: Leonard Sasse
"""

from kernelridgeclass import KernelRidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeClassifier

lambdas = [
    0,
    0.00001,
    0.0001,
    0.001,
    0.004,
    0.007,
    0.01,
    0.04,
    0.07,
    0.1,
    0.4,
    0.7,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    100,
    150,
    200,
    300,
    500,
    700,
    1000,
    10000,
    100000,
    1000000,
]

# Dict returning for each model the julearn name + params
model_dict_classification = {
    "kernelridge": [
        KernelRidgeClassifier(),
        {"kernelridgeclassifier__alpha": lambdas},
    ],
    "ridge": [
        RidgeClassifier(),
        {"ridgeclassifier__alpha": lambdas}
    ],
}

model_dict_regression = {
    "kernelridge": [KernelRidge(), {"kernelridge__alpha": lambdas}],
}


def get_model_grid(model_name, problem_type):
    if problem_type == "regression":
        model, param_dict = model_dict_regression[model_name]
    else:
        model, param_dict = model_dict_classification[model_name]

    print("----------------------------------------------------------------\n")
    print(f"The selected model is: {model}\n")
    print("Model Parameters are as follows:\n")
    for key, value in param_dict.items():
        print(f"{key} ==== {value}\n")
    print("----------------------------------------------------------------\n")
    return model, param_dict
