import argparse
import pandas as pd
from sklearn.metrics import classification_report
from typing import Tuple

"""
To run: 
After running python baseline.py -> bydefault would show
    the strict and loose evaluation for dictionary based model
python evaluate.py --goldpath "file path containing true
     labels" --predpath "filepath containing pred labels"
"""


def read_file(goldpath: str, predpath: str):
    """Reads the file and returns pandas dataframe

    Args:
        goldpath: csv filepath containing gold labels
        predpath: csv filepath containing predicted labels

    Returns:
        true_labels: pandas dataframe containing gold labels
        pred_labels: pandas dataframe containing predicted labels
    """
    true_labels = pd.read_csv(goldpath)
    pred_labels = pd.read_csv(predpath)

    return true_labels, pred_labels


def get_related_tags(true_labels, pred_labels):
    """
    Takes the dataframe and returns concatenated tags
    strict_gt = [[Admission, O], [Date, O], [eye, B-ADE, discharge, I-ADE], [the, O]]
    pred_gt = [[Admission, O], [Date, O], [eye, B-ADE, discharge, B-drug], [the, O]]

    Args:
        true_labels: pandas dataframe containing gold labels
        pred_labels: pandas dataframe containing predicted labels

    Returns:
        true_gt: modified list of list with all related entities of a tag together
        pred_gt: modified list of list with all related entities of a tag together
    """

    true_tags = true_labels[["token", "entity"]].values.tolist()
    pred_tags = pred_labels[["token", "argmax_entity"]].values.tolist()

    true_gt = []
    pred_gt = []

    j = 0
    for i in range(len(true_tags)):
        if true_tags[i][1].startswith("I-"):
            true_gt[j - 1].append(true_tags[i][0])
            true_gt[j - 1].append(true_tags[i][1])
            pred_gt[j - 1].append(pred_tags[i][0])
            pred_gt[j - 1].append(pred_tags[i][1])
        else:
            true_gt.append(true_tags[i])
            pred_gt.append(pred_tags[i])
            j += 1

    return true_gt, pred_gt


def get_predictions(true_gt, pred_gt):
    """
    Takes the concatenated related tokens and tags list and returns three list:
    ground truth, strict predictions and loose predictions

    Args:
        true_gt: list of list with all related entities of a tag together
        pred_gt: list of list with all related entities of a tag together

    Returns:
        gold_entity: list containing true labels
        pred_strict: list containing predicted labels calculated using strict evaluation
        pred_loose: list containing predicted labels calculated using loose evaluation
    """

    gold_entity = []
    pred_strict = []
    pred_loose = []

    # Loop through the length of predictions and find strict and loose predictions
    for i in range(len(true_gt)):
        # exact match
        if true_gt[i] == pred_gt[i]:
            if true_gt[i][1] != "O":
                pred_strict.append(true_gt[i][1].split("-")[1])
                pred_loose.append(true_gt[i][1].split("-")[1])
            else:
                pred_strict.append("O")
                pred_loose.append("O")

        else:
            # loose match
            if (
                len([i for i, j in zip(true_gt[i], pred_gt[i]) if i == j])
                >= (len(true_gt[i]) // 2) + 1
            ):
                if true_gt[i][1] != "O":
                    pred_loose.append(true_gt[i][1].split("-")[1])
                else:
                    pred_loose.append("O")

                if pred_gt[i][1] != "O":
                    pred_strict.append(pred_gt[i][1].split("-")[1])
                else:
                    pred_strict.append("O")

            else:
                # no match
                if pred_gt[i][1] != "O":
                    pred_strict.append(pred_gt[i][1].split("-")[1])
                    pred_loose.append(pred_gt[i][1].split("-")[1])
                else:
                    pred_strict.append("O")
                    pred_loose.append("O")

        if true_gt[i][1] != "O":
            gold_entity.append(true_gt[i][1].split("-")[1])
        else:
            gold_entity.append("O")

    return gold_entity, pred_strict, pred_loose


def main(args):
    true_labels, pred_labels = read_file(args.goldpath, args.predpath)
    true_gt, pred_gt = get_related_tags(true_labels, pred_labels)
    gold_entity, pred_strict, pred_loose = get_predictions(true_gt, pred_gt)

    print("----Strict Evaluation-----")
    print(classification_report(y_true=gold_entity, y_pred=pred_strict))
    print("----Loose Evaluation-----")
    print(classification_report(y_true=gold_entity, y_pred=pred_loose))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument(
        "--goldpath", type=str, default="test_true.csv", help=""
    )
    parser.add_argument(
        "--predpath", type=str, default="test_pred.csv", help=""
    )
    args = parser.parse_args()
    main(args)
