if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/scivocab_uncased.vocab
export BERT_WEIGHTS=$BERT_BASE_FOLDER/scibert_scivocab_uncased.tar.gz

export CONFIG_FILE=scirex_pwc/training_config/bert_entity_linking_bert.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

# export ERC_DATA_BASE_PATH=model_data/pwc_split_on_sectioned
export ERC_DATA_BASE_PATH=model_data/sciERC_in_pwc_format
export TRAIN_PATH=$ERC_DATA_BASE_PATH/train.jsonl
export DEV_PATH=$ERC_DATA_BASE_PATH/dev.jsonl
export TEST_PATH=$ERC_DATA_BASE_PATH/test.jsonl

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_linker/$1/}

python -m allennlp.run train -s $OUTPUT_BASE_PATH --include-package scirex_pwc --force $CONFIG_FILE