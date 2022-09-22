"""
Calculate metrics from an estimator
"""
import pandas as pd
import argparse

def metrics_report(TP=0, FP=0, TN=0, FN=0, printout=True):
    """
    Generate a metrics report from blind test results

    Parameters
    ----------

    TP : int
        True positives

    FP : int
        False positives

    TN : int
        True negatives

    FN : int
        False negatives

    """

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Calculate false positive rate
    FPR = FP / (FP + TN)
    # Calculate false negative rate
    FNR = FN / (FN + TP)
    # Calculate precision
    precision = TP / (TP + FP)
    # Calculate recall (sensitivity)
    recall = TP / (TP + FN)
    # Specificity
    specificity = TN / (TN + FP)
    # Calculate F1 score
    F1 = 2 * precision * recall / (precision + recall)

    metrics_dict = {
            "TP": [TP],
            "FP": [FP],
            "TN": [TN],
            "FN": [FN],
            "FPR": [FPR],
            "FNR": [FNR],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "specificity": [specificity],
            "F1": [F1],
            }

    df = pd.DataFrame.from_dict(metrics_dict)

    format_tup = (
            ".0f",
            ".0f",
            ".0f",
            ".0f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            )
    print(df.to_markdown(index=False,floatfmt=format_tup))


if __name__ == '__main__':
    """
    Generate a statistics report
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--printout", default=False, action="store_true",
            help="Flag to print out metrics to stdout.")
    parser.add_argument(
            "--TP", type=int, required=True,
            help="The true positive count.")
    parser.add_argument(
            "--FP", type=int, required=True,
            help="The false positive count.")
    parser.add_argument(
            "--TN", type=int, required=True,
            help="The true negative count.")
    parser.add_argument(
            "--FN", type=int, required=True,
            help="The false negative count.")

    # Collect arguments
    args = parser.parse_args()
    TP = args.TP
    FP = args.FP
    TN = args.TN
    FN = args.FN

    metrics_report(TP=TP, FP=FP, TN=TN, FN=FN)
