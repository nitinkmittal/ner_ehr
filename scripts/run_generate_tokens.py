"""This module provide CLI option to generate annotated and unannotated tokens
    from EHR annotations and EHR text-file respectively."""
import argparse
import glob
import os
import warnings
from pathlib import Path
from time import time
from typing import List, Union

import numpy as np
from ner_ehr.data.ehr import EHR
from ner_ehr.data.variables import AnnotationTuple, TokenTuple
from ner_ehr.tokenizers import (
    NLTKTokenizer,
    ScispacyTokenizer,
    SplitTokenizer,
    Tokenizer,
)
from ner_ehr.utils import save_kwargs, time_func
from tqdm import tqdm

from custom_parsers import CustomAnnotationParser, CustomTokenParser

DEFAULT_TOKENS_DATA_DIR_TRAIN: Union[Path, str] = os.path.join(
    os.getcwd(), "tokens", "train"
)
DEFAULT_TOKENS_DATA_DIR_VAL: Union[Path, str] = os.path.join(
    os.getcwd(), "tokens", "val"
)
DEFAULT_TOKENS_DATA_DIR_TEST: Union[Path, str] = os.path.join(
    os.getcwd(), "tokens", "test"
)
AVAILABLE_TOKENIZERS: List[str] = ["split", "nltk", "scispacy"]
DEFAULT_TOKENIZER: str = "nltk"
DEFAULT_SEP_FOR_SPLIT_TOKENIZER: str = " "
DEFAULT_VALIDATE_TOKEN_IDXS: str = "Y"
DEFAULT_VAL_SPLIT: float = 0.1
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_SAVE_PARSER_ARGS: str = "Y"
DEFAULT_PARSER_ARGS_SAVE_FP: Union[Path, str] = os.path.join(
    os.getcwd(),
    f"{os.path.basename(__file__).split('.')[0]}_{int(time())}"
    "_parser_args.yaml",
)


def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_data_dir",
        type=str,
        help=("directory with training EHR text and annotation files"),
    )

    parser.add_argument(
        "test_data_dir",
        type=str,
        help=("directory with testing EHR text and annotation files"),
    )

    parser.add_argument(
        "--val_split",
        type=float,
        help=(
            "validation split (from training data), "
            f"default: {DEFAULT_VAL_SPLIT}"
        ),
        default=DEFAULT_VAL_SPLIT,
    )

    parser.add_argument(
        "--tokens_dir_train",
        type=str,
        help=(
            "directory to store training tokens, "
            f"default: {DEFAULT_TOKENS_DATA_DIR_TRAIN}"
        ),
        default=DEFAULT_TOKENS_DATA_DIR_TRAIN,
    )

    parser.add_argument(
        "--tokens_dir_val",
        type=str,
        help=(
            "directory to store validation tokens, "
            f"default: {DEFAULT_TOKENS_DATA_DIR_VAL} "
        ),
        default=DEFAULT_TOKENS_DATA_DIR_VAL,
    )

    parser.add_argument(
        "--tokens_dir_test",
        type=str,
        help=(
            "directory to store test tokens, "
            f"default: {DEFAULT_TOKENS_DATA_DIR_TEST} "
        ),
        default=DEFAULT_TOKENS_DATA_DIR_TEST,
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        help=(
            "tokenizer to generate tokens, "
            f"available tokenizers: [{', '.join(AVAILABLE_TOKENIZERS)}], "
            f"default tokenizer: {DEFAULT_TOKENIZER}"
        ),
        default=DEFAULT_TOKENIZER,
    )

    parser.add_argument(
        "--sep",
        type=str,
        help=(
            "separator for split tokenizer if used, "
            f"default: `{DEFAULT_SEP_FOR_SPLIT_TOKENIZER}`"
        ),
        default=DEFAULT_SEP_FOR_SPLIT_TOKENIZER,
    )

    parser.add_argument(
        "--validate_token_idxs",
        type=str,
        help=(
            "should validate token start and end character indexes "
            " (sanity check) or not "
            f"(Y/N), default: {DEFAULT_VALIDATE_TOKEN_IDXS}"
        ),
        default=DEFAULT_VALIDATE_TOKEN_IDXS,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help=(
            f"random seed for reproducibility, default: {DEFAULT_RANDOM_SEED}"
        ),
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--save_parser_args",
        type=str,
        help=(
            "should save parser arguments or not (Y/N), "
            f"default: {DEFAULT_SAVE_PARSER_ARGS}"
        ),
        default=DEFAULT_SAVE_PARSER_ARGS,
    )

    parser.add_argument(
        "--parser_args_save_fp",
        type=str,
        help=(
            "filepath to save parser arguments, "
            "used if `save_parser_args` is set as `Y`, "
            f"default: {DEFAULT_PARSER_ARGS_SAVE_FP}"
        ),
        default=DEFAULT_PARSER_ARGS_SAVE_FP,
    )

    arguments = parser.parse_args()
    return arguments


@time_func
def main():
    args = parse_arguments()

    if args.save_parser_args == "Y":
        # saving parser arguments
        save_kwargs(fp=args.parser_args_save_fp, **args.__dict__)

    # initializing tokenizer
    validate_token_idxs = True if args.validate_token_idxs == "Y" else False
    if args.tokenizer == "nltk":
        tokenizer = NLTKTokenizer(validate_token_idxs=validate_token_idxs)
    elif args.tokenizer == "split":
        tokenizer = SplitTokenizer(
            sep=args.sep, validate_token_idxs=validate_token_idxs
        )
    elif args.tokenizer == "scispacy":
        tokenizer = ScispacyTokenizer(validate_token_idxs=validate_token_idxs)
    else:
        warnings.warn(
            f"`{args.tokenizer}` tokenizer not implemented, "
            f"using default `{DEFAULT_TOKENIZER}` tokenizer"
        )
        tokenizer = NLTKTokenizer(validate_token_idxs=validate_token_idxs)

    # appending type of tokenizer to names of tokens data dirs
    if args.tokenizer in AVAILABLE_TOKENIZERS:
        tokens_dir_append = f"_{args.tokenizer}"
    else:
        tokens_dir_append = f"_{DEFAULT_TOKENIZER}"

    # creating directories to store train, val and test data resp.
    # helps to identify type of tokenizer used
    tokens_dir_train = args.tokens_dir_train + tokens_dir_append
    os.makedirs(
        tokens_dir_train,
        exist_ok=True,
    )
    tokens_dir_val = args.tokens_dir_val + tokens_dir_append
    os.makedirs(tokens_dir_val, exist_ok=True)
    tokens_dir_test = args.tokens_dir_test + tokens_dir_append
    os.makedirs(tokens_dir_test, exist_ok=True)

    # getting all EHR text-files
    record_fps = glob.glob(os.path.join(args.train_data_dir, "*.txt"))

    # getting number of train and validation EHR text-files
    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(record_fps)
    num_train_record_fps = int((1 - args.val_split) * len(record_fps))

    # generating tokens from train records
    train_record_fps = record_fps[:num_train_record_fps]
    build_tokens(
        tokenizer=tokenizer,
        record_fps=train_record_fps,
        save_dir=tokens_dir_train,
    )
    # generating tokens from val records
    val_record_fps = record_fps[num_train_record_fps:]
    build_tokens(
        tokenizer=tokenizer,
        record_fps=val_record_fps,
        save_dir=tokens_dir_val,
    )
    # generating tokens from test records
    test_record_fps = glob.glob(os.path.join(args.test_data_dir, "*.txt"))
    build_tokens(
        tokenizer=tokenizer,
        record_fps=test_record_fps,
        save_dir=tokens_dir_test,
    )


def build_tokens(
    tokenizer: Tokenizer,
    record_fps: Union[Path, str],
    save_dir: Union[Path, str],
) -> None:
    """Utility function to generate unannotated and annotated tokens
    from given EHR text-files/records and corresponding EHR annotations
    and saving into CSVs.

    Args:
        tokenizer: initialized ner_utils.tokenizer.Tokenizer object

        record_fps: list of filepaths to EHR records/text-files

        save_dir: dir to dump/save annotated tuples
            if annotation are not available for tokens,
            then `O` (outside) entity is added by default for such tokens
    """
    annotation_parser = CustomAnnotationParser(tokenizer=tokenizer)
    token_parser = CustomTokenParser(tokenizer=tokenizer)
    ehr = EHR()

    for record_fp in tqdm(
        record_fps,
        leave=False,
        position=0,
    ):
        dir: Path = os.path.dirname(record_fp)  # record directory
        fp: str = os.path.basename(record_fp).split(".")[0]  # record filename

        # annotation filepath
        annotations_fp: Path = os.path.join(dir, f"{fp}.ann")
        # generating annotations
        annotations: List[AnnotationTuple] = annotation_parser.parse(
            annotations_fp=annotations_fp, record_fp=record_fp
        )

        # generating tokens
        tokens: List[TokenTuple] = token_parser.parse(
            record_fp=record_fp, annotations=annotations
        )

        # tokens record filepath
        tokens_record_fp: Path = os.path.join(save_dir, fp)

        # saving as CSVs
        ehr.write_csv_tokens_with_annotations(
            tokens=tokens, annotations=annotations, fp=tokens_record_fp
        )


if __name__ == "__main__":
    main()
