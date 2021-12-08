import argparse
import glob
import os
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from ner_ehr.data.ehr import EHR
from ner_ehr.data.variables import AnnotationTuple, TokenTuple
from ner_ehr.tokenizers import (NLTKTokenizer, ScispacyTokenizer,
                                SplitTokenizer, Tokenizer)
from ner_ehr.utils import save_kwargs
from tqdm import tqdm

from custom_parsers import CustomAnnotationParser, CustomTokenParser

DEFAULT_PROCESSED_DATA_DIR_TRAIN: Union[Path, str] = os.path.join(
    os.getcwd(), "processed", "train"
)
DEFAULT_PROCESSED_DATA_DIR_VAL: Union[Path, str] = os.path.join(
    os.getcwd(), "processed", "val"
)
DEFAULT_PROCESSED_DATA_DIR_TEST: Union[Path, str] = os.path.join(
    os.getcwd(), "processed", "test"
)

AVAILABLE_TOKENIZERS: List[str] = ["SPLIT", "NLTK", "SCISPACY"]
DEFAULT_TOKENIZER: str = "NLTK"
DEFAULT_SEP_FOR_SPLIT_TOKENIZER: str = " "
DEFAULT_VALIDATE_TOKEN_IDXS: str = "Y"
DEFAULT_VAL_SPLIT: float = 0.1
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_SAVE_PARSER_ARGS: str = "Y"
DEFAULT_PARSER_ARGS_SAVE_FP: Union[Path, str] = os.path.join(
    os.getcwd(), f"{os.path.basename(__file__).split('.')[0]}_parser_args.yaml"
)


def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_train_data_dir",
        type=str,
        help=("directory with training EHR txt and ann files"),
    )

    parser.add_argument(
        "input_test_data_dir",
        type=str,
        help=("directory with testing EHR txt and ann files"),
    )

    parser.add_argument(
        "--val_split",
        type=float,
        help=(
            "directory to store processed training tokens, "
            f"default: {DEFAULT_VAL_SPLIT}"
        ),
        default=DEFAULT_VAL_SPLIT,
    )

    parser.add_argument(
        "--processed_data_dir_train",
        type=str,
        help=(
            "directory to store processed training tokens, "
            f"default: {DEFAULT_PROCESSED_DATA_DIR_TRAIN}"
        ),
        default=DEFAULT_PROCESSED_DATA_DIR_TRAIN,
    )

    parser.add_argument(
        "--processed_data_dir_val",
        type=str,
        help=(
            "directory to store processed val tokens, "
            f"default: {DEFAULT_PROCESSED_DATA_DIR_VAL} "
        ),
        default=DEFAULT_PROCESSED_DATA_DIR_VAL,
    )

    parser.add_argument(
        "--processed_data_dir_test",
        type=str,
        help=(
            "directory to store processed test tokens, "
            f"default: {DEFAULT_PROCESSED_DATA_DIR_TEST} "
        ),
        default=DEFAULT_PROCESSED_DATA_DIR_TEST,
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        help=(
            "tokenizer to generate tokens, "
            f"available tokenizers: [{', '.join(AVAILABLE_TOKENIZERS)}]"
        ),
        default=DEFAULT_TOKENIZER,
    )

    parser.add_argument(
        "--sep",
        type=str,
        help=(
            "separator used only by split tokenizer, "
            f"default: {DEFAULT_SEP_FOR_SPLIT_TOKENIZER}"
        ),
        default=DEFAULT_SEP_FOR_SPLIT_TOKENIZER,
    )

    parser.add_argument(
        "--validate_token_idxs",
        type=str,
        help=(
            "should validate token character indexes (sanity check) or not "
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
            f"default: {DEFAULT_PARSER_ARGS_SAVE_FP}"
        ),
        default=DEFAULT_PARSER_ARGS_SAVE_FP,
    )

    arguments = parser.parse_args()
    return arguments


def main():
    args = parse_arguments()

    if args.save_parser_args == "Y":
        save_kwargs(fp=args.parser_args_save_fp, **args.__dict__)

    validate_token_idxs = True if args.validate_token_idxs == "Y" else False
    if args.tokenizer == "NLTK":
        tokenizer = NLTKTokenizer(validate_token_idxs=validate_token_idxs)
    elif args.tokenizer == "SPLIT":
        tokenizer = SplitTokenizer(
            sep=args.sep, validate_token_idxs=validate_token_idxs
        )
    elif args.tokenizer == "SCISPACY":
        tokenizer = ScispacyTokenizer(validate_token_idxs=validate_token_idxs)
    else:
        warnings.warn(
            f"{args.tokenizer} tokenizer not implemented, "
            f"using default {DEFAULT_TOKENIZER} tokenizer"
        )
        tokenizer = NLTKTokenizer(validate_token_idxs=validate_token_idxs)

    os.makedirs(args.processed_data_dir_train, exist_ok=True)
    os.makedirs(args.processed_data_dir_val, exist_ok=True)
    os.makedirs(args.processed_data_dir_test, exist_ok=True)

    record_fps = glob.glob(os.path.join(args.input_train_data_dir, "*.txt"))
    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(record_fps)
    num_train_record_fps = int((1 - args.val_split) * len(record_fps))

    # generating annotated tokens from train records
    train_record_fps = record_fps[:num_train_record_fps]
    build_processed_data(
        tokenizer=tokenizer,
        record_fps=train_record_fps,
        save_dir=args.processed_data_dir_train,
    )
    # generating annotated tokens from val records
    val_record_fps = record_fps[num_train_record_fps:]
    build_processed_data(
        tokenizer=tokenizer,
        record_fps=val_record_fps,
        save_dir=args.processed_data_dir_val,
    )
    # generating annotated tokens from test records
    test_record_fps = glob.glob(
        os.path.join(args.input_test_data_dir, "*.txt")
    )
    build_processed_data(
        tokenizer=tokenizer,
        record_fps=test_record_fps,
        save_dir=args.processed_data_dir_test,
    )


def build_processed_data(
    tokenizer: Tokenizer,
    record_fps: Union[Path, str],
    save_dir: Union[Path, str],
):
    """Utility function to generate tokens from given EHR records."""
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
        processed_record_fp: Path = os.path.join(
            save_dir, fp
        )  # processed record filepath
        annotations_fp: Path = os.path.join(
            dir, f"{fp}.ann"
        )  # annotation filepath

        # generating annotations
        annotations: List[AnnotationTuple] = annotation_parser.parse(
            annotations_fp=annotations_fp, record_fp=record_fp
        )

        # generating annotated tokens
        tokens: List[TokenTuple] = token_parser.parse(
            record_fp=record_fp, annotations=annotations
        )

        # saving as CSVs
        ehr.write_csv_tokens_with_annotations(
            tokens=tokens, annotations=annotations, fp=processed_record_fp
        )


if __name__ == "__main__":
    main()
