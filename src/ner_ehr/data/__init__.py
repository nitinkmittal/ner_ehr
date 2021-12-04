from enum import Enum


class Constants(Enum):
    UNTAG_ENTITY_LABEL = "O"  # outside entity
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN_IDX = 0
    UNK_TOKEN_IDX = 1
    UNTAG_ENTITY_INT_LABEL = 0
