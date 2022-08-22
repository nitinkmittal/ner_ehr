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
from ner_ehr.training.pl_models import LitLSTMCRFNERTagger, LitLSTMNERTagger
from ner_ehr.utils import read_annotatedtuples, save_np, time_func
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_TO_LOWER: str = "Y"
DEFAULT_IGNORE_RANDOM_TOKEN_PROB: float = 0.0
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
DEFAULT_BiLSTM: str = "N"
DEFAULT_NUM_LSTM_LAYERS: int = 1
DEFAULT_LSTM_DROPOUT: float = 0.0
DEFAULT_CRF: str = "N"
DEFAULT_MASKS: str = "N"
DEFAULT_LR: float = 0.001
DEFAULT_CE_WEIGHT: float = 1.0
DEFAULT_CRF_NLLH_WEIGHT: float = 0.001
DEFAULT_EPOCHS: int = 1
DEFAULT_SAVE_CM_AFTER_EVERY_N_EPOCHS: int = 1
DEFAULT_DEVICE_TYPE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
DEFAULT_NUM_GPUS: Optional[int] = (
    1 if DEFAULT_DEVICE_TYPE.type != "cpu" else None
)
DEFAULT_LOG_DIR: Union[Path, str] = os.path.join(os.getcwd(), "logs")
DEFAULT_EXPERIMENT_NAME: str = "ner_ehr_lstm"
DEFAULT_MODEL_CHECKPOINT_FP: int = "{epoch}-{step}-{val_loss:.3f}"
DEFAULT_AVAILABLE_MONITOR_AND_MODE_WITH_CRF = {
    "val_loss": "min",
    "val_ce_loss": "min",
    "val_crf_nllh": "min",
    "val_argmax_acc": "max",
    "val_viterbi_acc": "max",
}
DEFAULT_AVAILABLE_MONITOR_AND_MODE_WITHOUT_CRF = {
    "val_loss": "min",
    "val_argmax_acc": "max",
}
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
        "tokens_dir_train",
        type=str,
        help="directory containing training tokens",
    )

    parser.add_argument(
        "tokens_dir_val",
        type=str,
        help="directory containing validation tokens ",
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
        "--ignore_rand_token_prob",
        type=float,
        help=(
            "probability to ignore random training token "
            "while building vocab, "
            f"default: {DEFAULT_IGNORE_RANDOM_TOKEN_PROB}"
        ),
        default=DEFAULT_IGNORE_RANDOM_TOKEN_PROB,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        help=f"sequence length, default: {DEFAULT_MAX_SEQ_LENGTH}",
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
        "--bilstm",
        type=str,
        help=(
            "should use bidirectional lstm or not (Y/N), "
            f"default: {DEFAULT_BiLSTM}"
        ),
        default=DEFAULT_BiLSTM,
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
        "--crf",
        type=str,
        help=(
            "should use Conditional Random Field or not (Y/N), "
            f"default: {DEFAULT_CRF}"
        ),
        default=DEFAULT_CRF,
    )

    parser.add_argument(
        "--masks",
        type=str,
        help=(
            "should use masks when using Conditional Random Field "
            f" or not (Y/N), default: {DEFAULT_MASKS}"
        ),
        default=DEFAULT_MASKS,
    )

    parser.add_argument(
        "--ce_loss_weight",
        type=float,
        help=(
            "weight to cross entropy loss, " f" default: {DEFAULT_CE_WEIGHT}"
        ),
        default=DEFAULT_CE_WEIGHT,
    )

    parser.add_argument(
        "--crf_nllh_weight",
        type=float,
        help=(
            "weight to crf neg-log-likelihood loss, "
            f" default: {DEFAULT_CRF_NLLH_WEIGHT}"
        ),
        default=DEFAULT_CRF_NLLH_WEIGHT,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help=f"learning rate, default: {DEFAULT_LR}",
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help=f"number of training epochs, default: {DEFAULT_EPOCHS}",
        default=DEFAULT_EPOCHS,
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
        "--monitor",
        type=str,
        help=(
            "monitor criteria to save model checkpoint, "
            "available monitor criterias with crf: "
            f"[{', '.join(list(DEFAULT_AVAILABLE_MONITOR_AND_MODE_WITH_CRF.keys()))}], "
            f"without crf: "
            f"[{', '.join(list(DEFAULT_AVAILABLE_MONITOR_AND_MODE_WITHOUT_CRF.keys()))}], "
            f"default: {DEFAULT_MONITOR}"
        ),
        default=DEFAULT_MONITOR,
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
        dir_train=args.tokens_dir_train,
        dir_val=args.tokens_dir_val,
        to_lower=True if args.to_lower == "Y" else False,
        ignore_random_token_prob=args.ignore_rand_token_prob,
        seq_length=args.seq_len,
        embed_dim=args.embed_dim,
        pre_trained_embed_type=pre_trained_embed_type,
        load_pre_trained_embed_fp=load_pre_trained_embed_fp,
        save_pre_trained_embed_weights_fp=save_pre_trained_embed_weights_fp,
        batch_size_train=args.bs_train,
        batch_size_val=args.bs_val,
        num_workers_train=args.num_workers_train,
        num_workers_val=args.num_workers_val,
        hidden_size=args.hidden_size,
        bidirectional=True if args.bilstm == "Y" else False,
        num_lstm_layers=args.num_lstm_layers,
        lstm_dropout=args.lstm_dropout,
        crf=True if args.crf == "Y" else False,
        masks=True if args.masks == "Y" else False,
        ce_weight=args.ce_loss_weight,
        crf_nllh_weight=args.crf_nllh_weight,
        lr=args.lr,
        epochs=args.epochs,
        save_cm_after_every_n_epochs=args.save_cm_after_every_n_epochs,
        monitor=args.monitor,
        mode=DEFAULT_AVAILABLE_MONITOR_AND_MODE_WITH_CRF[args.monitor],
        log_dir=args.log_dir,
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
    """Load and save given pre-trained embeddings in `gensim.models.keyvectors`
        format and save embeddings vectors for tokens in given vocab.

    Note: loading pre-trained embeddings in `gensim.models.keyvectors`
        format is an expensive operation.

    Args:
        load_embed_fp: file-path to load pre-trained embeddings

        embed_type: type of pre-trained embeddings,
            support available for `glove` and `pubmed`

        unknown_token_embedding: A 2-D NumPy array of shape (embedding_dim, )
            embedding vector to be used for tokens not present
            in pre-trained embeddings vocab. Dimension of
            unknown embedding vector should be equal to
            dimension of pre-trained embedding vectors

        vocab: pre-trained ner_data.vocab.TokenEntityVocab object

        save_embed_weights_fp: file-path to pre-trained embeddings weights

    Returns:
        None
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

    # initializing embedding matrix
    embed_weights = np.zeros(
        (vocab.num_uniq_tokens, len(unknown_token_embedding)), dtype=np.float32
    )
    # loading embeddings for tokens in vocab
    for token, idx in tqdm(
        vocab._token_to_idx.items(), leave=False, position=0
    ):
        embed_weights[idx] = embed(tokens=token)[0]
    embed_weights = torch.tensor(embed_weights, dtype=torch.float32)
    logger.info(f"Saving embedding vectors as {save_embed_weights_fp}")
    save_np(arr=embed_weights, fp=save_embed_weights_fp)


@time_func
def run_model(
    dir_train: Union[str, Path],
    dir_val: Union[str, Path],
    to_lower: bool,
    ignore_random_token_prob: float,
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
    crf: bool,
    masks: bool,
    ce_weight: float,
    crf_nllh_weight: float,
    lr: float,
    epochs: int,
    save_cm_after_every_n_epochs: int,
    monitor: str,
    mode: str,
    log_dir: int,
    random_seed: int,
    parser_args: Dict[str, Any],
) -> None:
    """Function to load train annotatedtuples, build vocab,
    load embeddings(if required), train and validate model.

    Args:
        dir_train: directory containing CSVs with training
            annotated/unannotated tokens,
            if annotated, CSVs should have following columns:
                [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
            if unannotated, CSVs should have following columns:
                [`doc_id`, `token`, `start_idx`, `end_idx`]
            Usually annotated tokens are provided for training

        dir_val: directory containing CSVs with validation
            annotated/unannotated tokens,
            if annotated, CSVs should have following columns:
                [`doc_id`, `token`, `start_idx`, `end_idx`, `entity`]
            if unannotated, CSVs should have following columns:
                [`doc_id`, `token`, `start_idx`, `end_idx`]
            Usually annotated tokens are provided for validation

        to_lower: boolean flag
                if True, tokens are lowercased before adding into vocab,
                otherwise not

        ignore_random_token_prob: probability to ignore random token
            while building training vocab

        seq_length: maximum number of tokens in a sequence

        embed_dim: embedding dimension

        pre_trained_embed_type: optional, type of pre-trained embeddings,
            support available for `glove` and `pubmed`

        load_pre_trained_embed_fp: optional, file-path to load pre-trained
            embeddings

        save_pre_trained_embed_weights_fp: file-path to save pre-trained for
            vocab generated from training data

        batch_size_train: number of sequences of annotated
            tokens in a training batch

        batch_size_val: number of sequences of annotated
            tokens in a training batch

        num_workers_train: how many subprocesses to use for train
            data loading. 0 means that the data will be loaded in
            the main process

        num_workers_val: how many subprocesses to use for validation
            data loading. 0 means that the data will be loaded in
            the main process

        hidden_size: number of hidden units in LSTM layer

        bidirectional: If True, becomes a bidirectional LSTM

        num_lstm_layers: number of recurrent layers
            E.g., setting num_layers=2 would mean stacking two LSTMs
            together to form a stacked LSTM, with the second LSTM
            taking in outputs of the first LSTM and
            computing the final results

        lstm_dropout: if non-zero, introduces a dropout layer
            on the outputs of each LSTM layer except the last layer,
            with dropout probability equal to dropout

        crf: if True, LSTM with CRF is used, i.e
            weighted average of CRF negative log-likelihood and
            cross-entropy loss is used as loss metric

        masks: if True, masks are used in LSTM + CRF model, otherwise not.
            masks can be used to ignore particular entity labels

        ce_weight: scalar float value multiplied with cross-entropy loss

        crf_nllh_weight: scalar float value multiplied with
            CRF negative log-likelihood

        lr: learning rate

        epochs: number of training epochs

        save_cm_after_every_n_epochs: interval of epochs before saving
            training and validation confusion matrices

        monitor: monitor criteria to save model checkpoint,
            available criterias when using CRF with LSTM: [`val_loss`,
                `val_ce_loss`, `val_crf_nllh`,
                `val_argmax_acc`, `val_viterbi_acc`]
            available criterias when not using CRF with LSTM: [`val_loss`,
                `val_argmax_acc`]

        mode: monitor minimum or maximum of monitor criteria

        log_dir: logging directory

        random_seed: random seed for reproducibility

        parser_args: parser arguments

    Returns:
        None
    """

    logger.info("Loading training annotated tuples ...")
    train_annotatedtuples = read_annotatedtuples(dir=dir_train)

    logger.info("Building vocab from training annotated tuples ...")
    vocab = TokenEntityVocab(
        to_lower=to_lower,
        ignore_random_token_prob=ignore_random_token_prob,
        random_seed=random_seed,
    )
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
    # seeding for reproducibility
    pl.utilities.seed.seed_everything(random_seed)
    collate_fn_train = EHRBatchCollator(return_meta=False)
    collate_fn_val = EHRBatchCollator(return_meta=False)
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
    if not crf:
        model = LitLSTMNERTagger(
            embedding_dim=embed_dim,
            vocab_size=vocab.num_uniq_tokens,
            hidden_size=hidden_size,
            num_classes=vocab.num_uniq_entities,
            embedding_weights_fp=save_pre_trained_embed_weights_fp,
            num_lstm_layers=num_lstm_layers,
            bidirectional=bidirectional,
            lstm_dropout=lstm_dropout,
            lr=lr,
            save_cm_after_every_n_epochs=save_cm_after_every_n_epochs,
            parser_args=parser_args,
        )
    else:
        model = LitLSTMCRFNERTagger(
            embedding_dim=embed_dim,
            vocab_size=vocab.num_uniq_tokens,
            hidden_size=hidden_size,
            num_classes=vocab.num_uniq_entities,
            embedding_weights_fp=save_pre_trained_embed_weights_fp,
            num_lstm_layers=num_lstm_layers,
            bidirectional=bidirectional,
            lstm_dropout=lstm_dropout,
            use_masks=masks,
            ce_weight=ce_weight,
            crf_nllh_weight=crf_nllh_weight,
            lr=lr,
            save_cm_after_every_n_epochs=save_cm_after_every_n_epochs,
            parser_args=parser_args,
        )

    # logger for current experiment
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=DEFAULT_EXPERIMENT_NAME if crf is False else "ner_ehr_lstm_crf",
    )
    # save checkpoint callback during training
    save_checkpoint_callback = ModelCheckpoint(
        filename=DEFAULT_MODEL_CHECKPOINT_FP,
        monitor=monitor,
        mode=mode,
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

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
