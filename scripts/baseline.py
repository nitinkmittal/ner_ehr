import glob
from typing import List

import pandas as pd
from datasets.dataset import NerDataset
from datasets.vocab import TokenEntityVocab
from sklearn.metrics import classification_report


class BaseModel:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def train(self, dl: List):
        raise NotImplementedError("Not implemented..")

    def fit(self, dataset: List[List], model) -> (List, List):
        raise NotImplementedError("Not implemented..")

    def evaluate(self, y_gold: List, y_pred: List) -> dict:
        score = classification_report(y_gold, y_pred)
        return score


class Baseline(BaseModel):
    """Creates a baseline model of dictionary and returns the
    performance metrics on validation and test dataset"""

    def __init__(self):
        super().__init__(root_dir)

    def train(self, dl: List):
        raise NotImplementedError("Not implemented..")

    def fit(self, dataset: List[List], vocab: dict) -> (List, List):
        """
        Evaluates if the token from the given dataset is present in vocab and
        returns the gold label and predicted labels of the given dataset.

        Args:
            dataset: data in the form of list of list
            vocab: dictionary containing voab from training dataset
        Returns:
            y_gold: true labels of the dataset
            y_pred: predicted labels of the dataset
        """

        y_gold = []
        y_pred = []

        # check if token exists in the vocab, else assign it to 'O' tag
        for row in dataset:
            token = row[0]
            if token not in vocab:
                y_pred.append("O")
            else:
                y_pred.append(max(vocab[token], key=vocab[token].get))

            y_gold.append(row[-1])

        return y_gold, y_pred

    def evaluate(self, pred_labels: List, gold_labels: List) -> dict:
        """
        Given true and predicted labels, calculates the performance metrics

        Args:
            pred_labels: list of predicted tags
            gold_labels: list of actual tags
        Returns:
            score: report of all metrics like precision, recall, f1-score, micro_average and macro_average
            for all entity classes
        """

        labels = set(gold_labels)
        labels.remove("O")

        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        score = classification_report(
            gold_labels, pred_labels, labels=sorted_labels
        )
        # score = classification_report(gold_labels, pred_labels)
        return score


if __name__ == "__main__":

    root_dir = "../notebooks/out"
    baseline_model = Baseline()

    train = NerDataset(f"{root_dir}/train/")
    val = NerDataset(f"{root_dir}/val/", expand_vocab=False)
    test = NerDataset(f"{root_dir}/test/", expand_vocab=False)

    datasets = {"train": train, "val": val, "test": test}

    train_vocab = datasets["train"].vocab.stats
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    val_gold, val_pred = baseline_model.fit(val_ds, train_vocab)
    score_valid = baseline_model.evaluate(val_gold, val_pred)
    print(f"Validation metrics: {score_valid}")

    test_gold, test_pred = baseline_model.fit(test_ds, train_vocab)
    score_test = baseline_model.evaluate(test_gold, test_pred)
    print(f"Test metrics: {score_test}")
