1. Electronic-Health-Records(EHR) and associated annotations are present in form of free texts,
sample [EHR](https://github.com/nitinkmittal/ner_ehr/blob/lstm/data/train/100035.txt), associated [annotation](https://github.com/nitinkmittal/ner_ehr/blob/lstm/data/train/100035.ann).
Before training models, we need to generate **tokens** from EHR and associated annotations. To generate annotated/unannotated tokens from EHR(\*.txt) and associated annotation files(\*.ann) use script `run_generate_tokens.py`. This script provides CLI option along with optional arguments to generate tokens using different tokenizers. Also, our package will take care of perserving the start and end character-indexes of every token created in-order to reconstruct the original EHR record from tokens. 
```bash
conda activate ner_ehr
python run_generate_tokens.py data/train data/test \
  -- val_split .1 \ 
  --tokens_dir_train tokens/train \
  --tokens_dir_val tokens/val \
  --tokens_dir_test tokens/test \
  --validate_token_idxs Y
```

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
1. Parser to parse E