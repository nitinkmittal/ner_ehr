import os
from typing import List

import numpy as np
import pandas as pd
from ner_ehr.data import Constants
from ner_ehr.data.variables import AnnotationTuple
from ner_ehr.data.vocab import TokenEntityVocab
from ner_ehr.utils import read_annotatedtuples
from sklearn.metrics import classification_report
from tqdm import tqdm

"""
This file creates a baseline model based on dictionary and stores the labels in 2 files: test_true.csv and test_pred.csv
python baseline.py

To see the performance of dictionary based model. run
python evaluate.py 
(default params are given for the dictionary based model)
"""


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
                y_pred.append((token, Constants.UNTAG_ENTITY_LABEL.value))
            else:
                y_pred.append((token,
                    sorted(
                        pred_entity.items(), key=lambda x: x[1], reverse=True
                    )[0][0])
                )

            y_gold.append((token, row.entity))

        return y_gold, y_pred


if __name__ == "__main__":

    root_dir = "../tokens"

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
    #val_gold, val_pred = baseline.fit(val_annotatedtuples)
    #score_valid = baseline.evaluate(val_gold, val_pred)
    #print(f"Validation metrics: {score_valid}")s

    test_gold, test_pred = baseline.fit(test_annotatedtuples)
    gold_df = pd.DataFrame(test_gold, columns =['token', 'entity'])
    pred_df = pd.DataFrame(test_pred, columns =['token', 'argmax_entity'])

    gold_df.to_csv("test_true.csv", index=False)
    pred_df.to_csv("test_pred.csv", index=False)
    print("Predictions saved. Please run : python evaluate.py to see the result.")


