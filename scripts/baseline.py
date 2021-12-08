import glob
import os
from os import listdir
from os.path import isfile, join
from typing import List
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from ner_ehr.data import Constants
from ner_ehr.data.variables import AnnotationTuple
from ner_ehr.data.vocab import TokenEntityVocab
from utils import read_annotatedtuples


class Baseline:
    """Creates a baseline model of dictionary and returns the
    performance metrics on validation and test dataset"""

    def __init__(self, vocab: TokenEntityVocab):
      
        self.vocab = vocab

    def fit(self, dataset: List[AnnotationTuple]) -> (List, List):
        """
        Evaluates if the token from the given dataset is present in vocab and
        returns the gold label and predicted labels of the given dataset.

        Args:
            dataset: dataset in the form of list of annotation tuples
        Returns:
            y_gold: list of true labels
            y_pred: list of predicted labels
        """

        y_gold = []
        y_pred = []

        # check if token exists in the vocab, else assign it to 'O' tag
        for row in tqdm(dataset, leave=False, position=0):
            token = self.vocab._to_lower(row.token)
            pred_entity = self.vocab.token_entity_freq[token]
            if len(pred_entity) == 0:
                y_pred.append(Constants.UNTAG_ENTITY_LABEL.value)
            else:
                y_pred.append(sorted(pred_entity.items(), key=lambda x: x[1], reverse=True)[0][0])

            y_gold.append(row.entity)

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

        score = classification_report(gold_labels, pred_labels, labels=np.unique(gold_labels))
        return score


if __name__ == "__main__":

    root_dir = "../processed"

    # Reading files and creating vocab
    train_annotatedtuples = read_annotatedtuples(
        dir=os.path.join(root_dir, "train")
    )
    vocab = TokenEntityVocab(to_lower=True)
    vocab.fit(annotatedtuples=train_annotatedtuples)

    val_annotatedtuples = read_annotatedtuples(
        dir=os.path.join(root_dir, "val")
    )
    test_annotatedtuples = read_annotatedtuples(
        dir=os.path.join(root_dir, "test")
    )

    baseline = Baseline(vocab=vocab)

    # Performance on val and test data
    val_gold, val_pred = baseline.fit(val_annotatedtuples)
    score_valid = baseline.evaluate(val_gold, val_pred)
    print(f"Validation metrics: {score_valid}")

    test_gold, test_pred = baseline.fit(test_annotatedtuples)
    score_test = baseline.evaluate(test_gold, test_pred)
    print(f"Test metrics: {score_test}")
