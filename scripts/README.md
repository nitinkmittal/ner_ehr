# Instructions to run scripts

## Generating tokens

Electronic-Health-Records(EHR) and associated annotations are present in form of free texts,
sample [EHR](https://github.com/nitinkmittal/ner_ehr/blob/lstm/data/train/100035.txt), associated [annotation](https://github.com/nitinkmittal/ner_ehr/blob/lstm/data/train/100035.ann).
Before training models, we need to generate **tokens** from EHR and associated annotations. To generate annotated/unannotated tokens from EHR(\*.txt) and associated annotation files(\*.ann) use script `run_generate_tokens.py`. This script provides CLI option along with optional arguments to generate tokens using different tokenizers. Also, our package will take care of perserving the start and end character-indexes of every token created in-order to reconstruct the original EHR record from tokens. 

```bash
python run_generate_tokens.py data/train data/test \
  --val_split .1 \ 
  --tokens_dir_train tokens/train \
  --tokens_dir_val tokens/val \
  --tokens_dir_test tokens/test \
  --validate_token_idxs Y
```

### Usage for `run_generate_tokens.py` 
```bash
python run_generate_tokens.py -h

usage: run_generate_tokens.py [-h] [--val_split VAL_SPLIT] [--tokens_dir_train TOKENS_DIR_TRAIN] [--tokens_dir_val TOKENS_DIR_VAL] [--tokens_dir_test TOKENS_DIR_TEST] [--tokenizer TOKENIZER] [--sep SEP] [--validate_token_idxs VALIDATE_TOKEN_IDXS]
                              [--random_seed RANDOM_SEED] [--save_parser_args SAVE_PARSER_ARGS] [--parser_args_save_fp PARSER_ARGS_SAVE_FP]
                              train_data_dir test_data_dir

positional arguments:
  train_data_dir        directory with training EHR text and annotation files
  test_data_dir         directory with testing EHR text and annotation files

optional arguments:
  -h, --help            show this help message and exit
  --val_split VAL_SPLIT
                        validation split (from training data), default: 0.1
  --tokens_dir_train TOKENS_DIR_TRAIN
                        directory to store training tokens, default: tokens/train
  --tokens_dir_val TOKENS_DIR_VAL
                        directory to store validation tokens, default: tokens/val
  --tokens_dir_test TOKENS_DIR_TEST
                        directory to store test tokens, default: tokens/test
  --tokenizer TOKENIZER
                        tokenizer to generate tokens, available tokenizers: [split, nltk, scispacy], default tokenizer: nltk
  --sep SEP             separator for split tokenizer if used, default: ` `
  --validate_token_idxs VALIDATE_TOKEN_IDXS
                        should validate token start and end character indexes (sanity check) or not (Y/N), default: Y
  --random_seed RANDOM_SEED
                        random seed for reproducibility, default: 42
  --save_parser_args SAVE_PARSER_ARGS
                        should save parser arguments or not (Y/N), default: Y
  --parser_args_save_fp PARSER_ARGS_SAVE_FP
                        filepath to save parser arguments, used if `save_parser_args` is set as `Y`, default: run_generate_tokens_{current_time}_parser_args.yaml
```

## Training rule-based baseline and evaluating

To train rule based model using generated training and validation tokens, use script `baseline.py`. This script will generate two csvs and will save it in the same folder where the script is placed.

### To train baseline model:
```bash
python baseline.py
```

To evaluate any model, use script 'evaluate.py'. Default arguments are set for the baseline model.
### To evaluate models:
```bash
python evaluate.py --goldpath 'csvfile path containing true labels' --predpath 'csvfile path containing predicted labels'
```

## Training LSTMs with and without Conditional Random Field(CRF)

To train LSTMs and BiLSTMs with and without CRF using generated training and validation tokens, use script `run_train_lstm.py`.

### To train LSTM:
```bash
python run_train_lstm.py ../tokens/train_scispacy ../tokens/val_scispacy
```

### To train BiLSTM:
```bash
python run_train_lstm.py ../tokens/train_scispacy ../tokens/val_scispacy \
  --bilstm Y
```

### To train LSTM/BiLSTM with CRF:
```bash
python run_train_lstm.py ../tokens/train_scispacy ../tokens/val_scispacy \
  --bilstm Y \
  --crf Y
```

### Usage for `run_train_lstm.py`
```bash
python run_train_lstm.py --help

usage: run_train_lstm.py [-h] [--to_lower TO_LOWER] [--seq_len SEQ_LEN] [--embed_dim EMBED_DIM] [--use_pre_trained_embed USE_PRE_TRAINED_EMBED]
                         [--pre_trained_embed_type PRE_TRAINED_EMBED_TYPE] [--load_pre_trained_embed_fp LOAD_PRE_TRAINED_EMBED_FP]
                         [--save_pre_trained_embed_weights_fp SAVE_PRE_TRAINED_EMBED_WEIGHTS_FP] [--bs_train BS_TRAIN] [--bs_val BS_VAL] [--num_workers_train NUM_WORKERS_TRAIN]
                         [--num_workers_val NUM_WORKERS_VAL] [--hidden_size HIDDEN_SIZE] [--bilstm BILSTM] [--num_lstm_layers NUM_LSTM_LAYERS] [--lstm_dropout LSTM_DROPOUT] [--crf CRF]
                         [--masks MASKS] [--ce_loss_weight CE_LOSS_WEIGHT] [--crf_nllh_weight CRF_NLLH_WEIGHT] [--lr LR] [--epochs EPOCHS]
                         [--save_cm_after_every_n_epochs SAVE_CM_AFTER_EVERY_N_EPOCHS] [--monitor MONITOR] [--log_dir LOG_DIR] [--random_seed RANDOM_SEED]
                         tokens_dir_train tokens_dir_val

positional arguments:
  tokens_dir_train      directory containing training tokens
  tokens_dir_val        directory containing validation tokens

optional arguments:
  -h, --help            show this help message and exit
  --to_lower TO_LOWER   should lowercase tokens or not while building training-vocab and pre-trained embeddings (if specified) default: Y
  --seq_len SEQ_LEN     sequence length, default: 256
  --embed_dim EMBED_DIM
                        embedding dimension, default: 50
  --use_pre_trained_embed USE_PRE_TRAINED_EMBED
                        should use pre-trained embeddings or not (Y/N), default: N
  --pre_trained_embed_type PRE_TRAINED_EMBED_TYPE
                        if `use_pre_trained_embed`=`Y`, specify type of pre-trained embeddings from available pre-trained embeddings: [glove, pubmed], default: glove
  --load_pre_trained_embed_fp LOAD_PRE_TRAINED_EMBED_FP
                        if `use_pre_trained_embed`=`Y`, specify filepath for pre-trained embeddings, default: /home/mittal.nit/projects/ner_ehr/scripts/glove.6B.50d.txt
  --save_pre_trained_embed_weights_fp SAVE_PRE_TRAINED_EMBED_WEIGHTS_FP
                        if `use_pre_trained_embed`=`Y`, specify filepath to save pre-trained embedding vectors, default: /home/mittal.nit/projects/ner_ehr/scripts/embedding_weights.npy
  --bs_train BS_TRAIN   training batch-size, default: 32
  --bs_val BS_VAL       validation batch-size, default: 32
  --num_workers_train NUM_WORKERS_TRAIN
                        number of workers for training dataloader, default: 9
  --num_workers_val NUM_WORKERS_VAL
                        number of workers for validation dataloader, default: 5
  --hidden_size HIDDEN_SIZE
                        number of hidden units in each lstm layer, default: 64
  --bilstm BILSTM       should use bidirectional lstm or not (Y/N), default: N
  --num_lstm_layers NUM_LSTM_LAYERS
                        number of stacked lstm layers, default: 1
  --lstm_dropout LSTM_DROPOUT
                        dropout for lstm layers, default: 0.0
  --crf CRF             should use Conditional Random Field or not (Y/N), default: N
  --masks MASKS         should use masks when using Conditional Random Field or not (Y/N), default: N
  --ce_loss_weight CE_LOSS_WEIGHT
                        weight to cross entropy loss, default: 1.0
  --crf_nllh_weight CRF_NLLH_WEIGHT
                        weight to crf neg-log-likelihood loss, default: 0.001
  --lr LR               learning rate, default: 0.001
  --epochs EPOCHS       number of training epochs, default: 1
  --save_cm_after_every_n_epochs SAVE_CM_AFTER_EVERY_N_EPOCHS
                        number of training epochs before saving a confusion matrix , default: 1
  --monitor MONITOR     monitor criteria to save model checkpoint, available monitor criterias with crf: [val_loss, val_ce_loss, val_crf_nllh, val_argmax_acc, val_viterbi_acc], without
                        crf: [val_loss, val_argmax_acc], default: val_loss
  --log_dir LOG_DIR     logging directory, default: /home/mittal.nit/projects/ner_ehr/scripts/logs
  --random_seed RANDOM_SEED
                        random seed for reproducibility, default: 42
```