"""
Logistic Regression
"""

import os
import glob 
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


# Labels:
# 0 : Normal
# 1 : Cancer

def logistic_regression(dataframe):
    """
    Performs logistic regression on a pandas ``DataFrame``

    Parameters
    ----------
    df : pd.DataFrame
        Pandas ``DataFrame`` object

    """
    return

if __name__ == '__main__':
    """
    Run logistic regression on a data set
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--csv_path", default=None, required=True,
            help="The path containing csv with data to perform logistic regression on")

    # Collect arguments
    args = parser.parse_args()
    csv_path = args.input_path

    # Read csv into pandas
    df = pd.read_csv(csv_path)

    # Perform logistic regression on the dataframe
    results = logistic_regression(df)
