"""This module contains all constants used throughout ner_ehr package."""
from enum import Enum


class Constants(Enum):
    # pad token, token idx, it's string entity and it's int entity label,
    #   pad token is considered as `garbage` token
    PAD_TOKEN = "<PAD>"
    PAD_TOKEN_ENTITY_LABEL = "P"
    PAD_TOKEN_ENTITY_INT_LABEL = 0
    PAD_TOKEN_IDX = 0

    UNK_TOKEN = "<UNK>"
    UNK_TOKEN_IDX = 1

    # outside entity and it's int label
    UNTAG_ENTITY_LABEL = "O"
    UNTAG_ENTITY_INT_LABEL = 1
