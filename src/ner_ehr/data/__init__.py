from enum import Enum


class Constants(Enum):
    PAD_TOKEN = "<PAD>"
    PAD_TOKEN_ENTITY_LABEL = "P"  # garbage entity
    PAD_TOKEN_ENTITY_INT_LABEL = 0  # garbage entity label
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN_IDX = 0
    UNK_TOKEN_IDX = 1
    UNTAG_ENTITY_LABEL = "O"  # outside entity
    UNTAG_ENTITY_INT_LABEL = 1  # outside entity label
