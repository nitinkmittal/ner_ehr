import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from ner_ehr.data.embeddings import GloveEmbeddings, PubMedicalEmbeddings
from ner_ehr.data.vocab import TokenEntityVocab
from ner_ehr.training.datasets import EHRBatchCollator, EHRDataModule
from ner_ehr.training.models import LitLSTMNERTagger
from ner_ehr.utils import save_np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from utils import read_annotatedtuples

logger = logging.getLogger(__name__)

DEFAULT_TO_LOWER: str = "Y"
DEFAULT_MAX_SEQ_LENGTH: int = 256
DEFAULT_EMBED_DIM: int = 50
DEFAULT_USE_PRE_TRAINED_EMBED: str = "N"
DEFAULT_AVAILABLE_PRE_TRAINED_EMBED_TYPES: List[str] = [
    "glove",
    "pubmed",
]
DEFAULT_PRE_TRAINED_EMBED: str = "glove"
DEFAULT_LOAD_PRE_TRAINED_EMBED_PATH: Union[Path, str] = os.path.join(
    os.getcwd(), f"glove.6B.{DEFAULT_EMBED_DIM}d.txt"
)
DEFAULT_SAVE_PRE_TRAINED_EMBED_WEIGHTS_PATH: Union[Path, str] = os.path.join(
    os.getcwd(), "embedding_weights.npy"
)

DEFAULT_BATCH_SIZE_TRAIN: int = 32
DEFAULT_BATCH_SIZE_VAL: int = 32
NUM_WORKER_CPUs = os.cpu_count() // 2
DEFAULT_NUM_WORKERS_TRAIN: int = max(1, int(NUM_WORKER_CPUs // 1.5))
DEFAULT_NUM_WORKERS_VAL: int = max(
    1, NUM_WORKER_CPUs - DEFAULT_NUM_WORKERS_TRAIN
)

DEAULT_HIDDEN_SIZE: int = 64
DEFAULT_USE_BIDIRECTIONAL_LSTM: str = "N"
DEFAULT_NUM_LSTM_LAYERS: int = 1
DEFAULT_LSTM_DROPOUT: float = 0.1

DEFAULT_LR: float = 0.001
DEFAULT_NUM_EPOCHS: int = 1
DEFAULT_SAVE_CM_AFTER_EVERY_N_EPOCHS: int = 1
DEFAULT_DEVICE_TYPE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
DEFAULT_NUM_GPUS: Optional[int] = (
    1 if DEFAULT_DEVICE_TYPE.type != "cpu" else None
)

DEFAULT_LOG_DIR: Union[Path, str] = os.path.join(os.getcwd(), "logs")
DEFAULT_EXPERIMENT_NAME = "ner_ehr_lstm"
DEFAULT_MODEL_CHECKPOINT_FP: int = "{epoch}-{step}-{val_loss:.3f}"

DEFAULT_MONITOR: str = "val_loss"
DEFAULT_MODE: str = "min"
DEFAULT_SAVE_TOP_K: int = 1
DEFAULT_NUM_SANITY_VAL_STEPS: int = 1
DEFAULT_LOG_EVERY_N_STEPS: int = 1
DEFAULT_RANDOM_SEED: int = 42


def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "processed_data_dir_train",
        type=str,
        help="directory to store processed training tokens",
    )

    parser.add_argument(
        "processed_data_dir_val",
        type=str,
        help="directory to store processed val tokens ",
    )

    parser.add_argument(
        "--to_lower",
        type=str,
        help=(
            "should lowercase tokens or not while building training-vocab "
            "and pre-trained embeddings (if specified) "
            f"default: {DEFAULT_TO_LOWER}"
        ),
        default=DEFAULT_TO_LOWER,
    )

    parser.add_argument(
        "--max_seq_len",
        type=int,
        help=f"maximum sequence length, default: {DEFAULT_MAX_SEQ_LENGTH}",
        default=DEFAULT_MAX_SEQ_LENGTH,
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        help=f"embedding dimension, default: {DEFAULT_EMBED_DIM}",
        default=DEFAULT_EMBED_DIM,
    )

    parser.add_argument(
        "--use_pre_trained_embed",
        type=str,
        help=(
            "should use pre-trained embeddings or not (Y/N), "
            f"default: {DEFAULT_USE_PRE_TRAINED_EMBED}"
        ),
        default=DEFAULT_USE_PRE_TRAINED_EMBED,
    )

    parser.add_argument(
        "--pre_trained_embed_type",
        type=str,
        help=(
            "if `use_pre_trained_embed`=`Y`, "
            "specify type of pre-trained embeddings from "
            "available pre-trained embeddings: "
            f"[{', '.join(DEFAULT_AVAILABLE_PRE_TRAINED_EMBED_TYPES)}], "
            f"default: {DEFAULT_PRE_TRAINED_EMBED}"
        ),
        default=DEFAULT_PRE_TRAINED_EMBED,
    )

    parser.add_argument(
        "--load_pre_trained_embed_fp",
        type=str,
        help=(
            "if `use_pre_trained_embed`=`Y`, "
            "specify filepath for pre-trained embeddings, "
            f"default: {DEFAULT_LOAD_PRE_TRAINED_EMBED_PATH}"
        ),
        default=DEFAULT_LOAD_PRE_TRAINED_EMBED_PATH,
    )

    parser.add_argument(
        "--save_pre_trained_embed_weights_fp",
        type=str,
        help=(
            "if `use_pre_trained_embed`=`Y`, "
            "specify filepath to save pre-trained embedding vectors, "
            f"default: {DEFAULT_SAVE_PRE_TRAINED_EMBED_WEIGHTS_PATH}"
        ),
        default=DEFAULT_SAVE_PRE_TRAINED_EMBED_WEIGHTS_PATH,
    )

    parser.add_argument(
        "--bs_train",
        type=int,
        help=f"training batch-size, default: {DEFAULT_BATCH_SIZE_TRAIN}",
        default=DEFAULT_BATCH_SIZE_TRAIN,
    )

    parser.add_argument(
        "--bs_val",
        type=int,
        help=f"validation batch-size, default: {DEFAULT_BATCH_SIZE_VAL}",
        default=DEFAULT_BATCH_SIZE_VAL,
    )

    parser.add_argument(
        "--num_workers_train",
        type=int,
        help=(
            "number of workers for training dataloader, "
            f"default: {DEFAULT_NUM_WORKERS_TRAIN}"
        ),
        default=DEFAULT_NUM_WORKERS_TRAIN,
    )

    parser.add_argument(
        "--num_workers_val",
        type=int,
        help=(
            "number of workers for validation dataloader, "
            f"default: {DEFAULT_NUM_WORKERS_VAL}"
        ),
        default=DEFAULT_NUM_WORKERS_VAL,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        help=(
            "number of hidden units in each lstm layer, "
            f"default: {DEAULT_HIDDEN_SIZE}"
        ),
        default=DEAULT_HIDDEN_SIZE,
    )

    parser.add_argument(
        "--use_bilstm",
        type=str,
        help=(
            "should use bidirectional lstm or not (Y/N), "
            f"default: {DEFAULT_USE_BIDIRECTIONAL_LSTM}"
        ),
        default=DEFAULT_USE_BIDIRECTIONAL_LSTM,
    )

    parser.add_argument(
        "--num_lstm_layers",
        type=int,
        help=(
            "number of stacked lstm layers, "
            f"default: {DEFAULT_NUM_LSTM_LAYERS}"
        ),
        default=DEFAULT_NUM_LSTM_LAYERS,
    )

    parser.add_argument(
        "--lstm_dropout",
        type=float,
        help=f"dropout for lstm layers, default: {DEFAULT_LSTM_DROPOUT}",
        default=DEFAULT_LSTM_DROPOUT,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help=f"learning rate, default: {DEFAULT_LR}",
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help=f"number of training epochs, default: {DEFAULT_NUM_EPOCHS}",
        default=DEFAULT_NUM_EPOCHS,
    )

    parser.add_argument(
        "--save_cm_after_every_n_epochs",
        type=int,
        help=(
            "number of training epochs before saving a confusion matrix "
            f", default: {DEFAULT_SAVE_CM_AFTER_EVERY_N_EPOCHS}"
        ),
        default=DEFAULT_SAVE_CM_AFTER_EVERY_N_EPOCHS,
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        help=f"logging directory, default: {DEFAULT_LOG_DIR}",
        default=DEFAULT_LOG_DIR,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help=(
            f"random seed for reproducibility, default: {DEFAULT_RANDOM_SEED}"
        ),
        default=DEFAULT_RANDOM_SEED,
    )

    arguments = parser.parse_args()
    return arguments


def main():

    args = parse_arguments()
    if (
        args.use_pre_trained_embed == "Y"
        and args.pre_trained_embed_type
        not in DEFAULT_AVAILABLE_PRE_TRAINED_EMBED_TYPES
    ):
        warnings.warn(
            f"Given pre-trained embeddings `{args.pre_trained_embed_type}` "
            "not available, setting embedding layer to be trainable"
        )
        pre_trained_embed_type = None
        load_pre_trained_embed_fp = None
        save_pre_trained_embed_weights_fp = None
    elif args.use_pre_trained_embed == "N":
        pre_trained_embed_type = None
        load_pre_trained_embed_fp = None
        save_pre_trained_embed_weights_fp = None
    else:
        pre_trained_embed_type = args.pre_trained_embed_type
        load_pre_trained_embed_fp = args.load_pre_trained_embed_fp
        save_pre_trained_embed_weights_fp = (
            args.save_pre_trained_embed_weights_fp
        )

    logger.info("Initializing environment for model training...")
    run_model(
        dir_train=args.processed_data_dir_train,
        dir_val=args.processed_data_dir_val,
        to_lower=True if args.to_lower == "Y" else False,
        seq_length=args.max_seq_len,
        embed_dim=args.embed_dim,
        pre_trained_embed_type=pre_trained_embed_type,
        load_pre_trained_embed_fp=load_pre_trained_embed_fp,
        save_pre_trained_embed_weights_fp=save_pre_trained_embed_weights_fp,
        batch_size_train=args.bs_train,
        batch_size_val=args.bs_val,
        num_workers_train=args.num_workers_train,
        num_workers_val=args.num_workers_val,
        hidden_size=args.hidden_size,
        bidirectional=True if args.use_bilstm == "Y" else False,
        num_lstm_layers=args.num_lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lr=args.lr,
        epochs=args.num_epochs,
        save_cm_after_every_n_epochs=args.save_cm_after_every_n_epochs,
        log_dir=args.log_dir,
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        random_seed=args.random_seed,
        parser_args=args.__dict__,
    )


def load_and_save_pre_trained_embed(
    load_embed_fp: Union[Path, str],
    embed_type: str,
    unknown_token_embedding: np.ndarray,
    vocab: TokenEntityVocab,
    save_embed_weights_fp: Union[Path, str],
) -> None:
    """To load given pre-trained embeddings in `gensim.models.keyvectors`
        format and save embeddings vectors for tokens in given vocab.

    Note: loading pre-trained embeddings in `gensim.models.keyvectors`
        format is an expensive operation.
    """
    logger.info(
        f"Loading pre-trained `{embed_type}` embeddings "
        f"from {load_embed_fp}"
    )

    if embed_type == "glove":
        embed = GloveEmbeddings(
            unknown_token_embedding=unknown_token_embedding,
            glove_fp=load_embed_fp,
            to_lower=vocab.to_lower,
        )
    else:
        embed = PubMedicalEmbeddings(
            unknown_token_embedding=unknown_token_embedding,
            pubmed_fp=load_embed_fp,
            to_lower=vocab.to_lower,
        )

    embed.load_word2vec()

    embed_weights = np.zeros(
        (vocab.num_uniq_tokens, len(unknown_token_embedding)), dtype=np.float32
    )
    for token, idx in tqdm(
        vocab._token_to_idx.items(), leave=False, position=0
    ):
        embed_weights[idx] = embed(tokens=token)[0]
    embed_weights = torch.tensor(embed_weights, dtype=torch.float32)
    logger.info(f"Saving embedding vectors as {save_embed_weights_fp}")
    save_np(arr=embed_weights, fp=save_embed_weights_fp)


def run_model(
    dir_train: Union[str, Path],
    dir_val: Union[str, Path],
    to_lower: bool,
    seq_length: int,
    embed_dim: int,
    pre_trained_embed_type: Optional[str],
    load_pre_trained_embed_fp: Optional[Union[Path, str]],
    save_pre_trained_embed_weights_fp: Optional[Union[Path, str]],
    batch_size_train: int,
    batch_size_val: int,
    num_workers_train: int,
    num_workers_val: int,
    hidden_size: int,
    bidirectional: bool,
    num_lstm_layers: int,
    lstm_dropout: float,
    lr=lr,
    epochs: int,
    save_cm_after_every_n_epochs: int,
    log_dir: int,
    experiment_name: str,
    random_seed: int,
    parser_args: Dict[str, Any],
):
    """Function to load train annotatedtuples, build vocab,
    load embeddings(if required), train and validate model."""
    # seeding for reproducibility
    pl.utilities.seed.seed_everything(random_seed)

    logger.info("Loading training annotated tuples ...")
    train_annotatedtuples = read_annotatedtuples(dir=dir_train)

    logger.info("Building vocab from training annotated tuples ...")
    vocab = TokenEntityVocab(to_lower=to_lower)
    vocab.fit(annotatedtuples=train_annotatedtuples)

    # loading and saving pre-trained embeddings if required
    if load_pre_trained_embed_fp is not None:
        rng = np.random.default_rng(random_seed)
        load_and_save_pre_trained_embed(
            load_embed_fp=load_pre_trained_embed_fp,
            embed_type=pre_trained_embed_type,
            unknown_token_embedding=rng.normal(embed_dim),
            vocab=vocab,
            save_embed_weights_fp=save_pre_trained_embed_weights_fp,
        )

    logger.info("Initializing PyTorch Lightning data module ...")
    collate_fn_train = EHRBatchCollator(return_meta=False)
    collate_fn_val = EHRBatchCollator(return_meta=True)
    datamodule = EHRDataModule(
        vocab=vocab,
        seq_length=seq_length,
        annotated=True,
        dir_train=dir_train,
        dir_val=dir_val,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        num_workers_train=num_workers_train,
        num_workers_val=num_workers_val,
        collate_fn_train=collate_fn_train,
        collate_fn_val=collate_fn_val,
    )

    logger.info("Initializing PyTorch Lightning Model ...")
    lit_lstm = LitLSTMNERTagger(
        embedding_dim=embed_dim,
        vocab_size=vocab.num_uniq_tokens,
        hidden_size=hidden_size,
        num_classes=vocab.num_uniq_entities,
        embedding_weights_fp=save_pre_trained_embed_weights_fp,
        num_lstm_layers=num_lstm_layers,
        bidirectional=bidirectional,
        lstm_dropout=lstm_dropout,
        save_cm_after_every_n_epochs=save_cm_after_every_n_epochs,
        lr=lr,
        parser_args=parser_args,
    )

    # logger for current experiment
    csv_logger = CSVLogger(save_dir=log_dir, name=experiment_name)
    # save checkpoint callback during training
    save_checkpoint_callback = ModelCheckpoint(
        filename=DEFAULT_MODEL_CHECKPOINT_FP,
        monitor=DEFAULT_MONITOR,
        mode=DEFAULT_MODE,
        save_top_k=DEFAULT_SAVE_TOP_K,
    )

    logger.info("Initializing PyTorch Lightning trainer ...")
    trainer = Trainer(
        gpus=DEFAULT_NUM_GPUS,
        max_epochs=epochs,
        log_every_n_steps=DEFAULT_LOG_EVERY_N_STEPS,
        logger=csv_logger,
        callbacks=[save_checkpoint_callback],
        num_sanity_val_steps=DEFAULT_NUM_SANITY_VAL_STEPS,
    )

    trainer.fit(model=lit_lstm, datamodule=datamodule)


if __name__ == "__main__":
    main()