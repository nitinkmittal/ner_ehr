usage: run_build_data.py [-h] [--val_split VAL_SPLIT] [--processed_data_dir_train PROCESSED_DATA_DIR_TRAIN] [--processed_data_dir_val PROCESSED_DATA_DIR_VAL]
                         [--processed_data_dir_test PROCESSED_DATA_DIR_TEST] [--tokenizer TOKENIZER] [--sep SEP] [--validate_token_idxs VALIDATE_TOKEN_IDXS] [--random_seed RANDOM_SEED]
                         [--save_parser_args SAVE_PARSER_ARGS] [--parser_args_save_fp PARSER_ARGS_SAVE_FP]
                         input_train_data_dir input_test_data_dir

positional arguments:
  input_train_data_dir  directory with training EHR text and annotation files
  input_test_data_dir   directory with testing EHR text and annotation files

optional arguments:
  -h, --help            show this help message and exit
  --val_split VAL_SPLIT
                        validation split (from training data), default: 0.1
  --processed_data_dir_train PROCESSED_DATA_DIR_TRAIN
                        directory to store processed training tokens, default: /home/mittal.nit/projects/ner_ehr/scripts/processed/train
  --processed_data_dir_val PROCESSED_DATA_DIR_VAL
                        directory to store processed validation tokens, default: /home/mittal.nit/projects/ner_ehr/scripts/processed/val
  --processed_data_dir_test PROCESSED_DATA_DIR_TEST
                        directory to store processed test tokens, default: /home/mittal.nit/projects/ner_ehr/scripts/processed/test
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
                        filepath to save parser arguments, used if `save_parser_args` is set as `Y`, default:
                        /home/mittal.nit/projects/ner_ehr/scripts/run_build_data_1639247685_parser_args.yaml
