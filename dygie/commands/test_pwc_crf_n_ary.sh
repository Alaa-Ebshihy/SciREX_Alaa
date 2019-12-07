if [ $# -eq  0 ]
  then
    echo "No argument supplied for experiment name"
    exit 1
fi

export BERT_VOCAB=$BERT_BASE_FOLDER/scivocab_uncased.vocab
export BERT_WEIGHTS=$BERT_BASE_FOLDER/scibert_scivocab_uncased.tar.gz

export CONFIG_FILE=dygie/training_config/pwc_config_crf_n_ary.jsonnet

export CUDA_DEVICE=$CUDA_DEVICE

export IS_LOWERCASE=true

export DATA_BASE_PATH=model_data/dataset_readers_paths

export TRAIN_DATASETS=pwc
export TRAIN_PATH=$DATA_BASE_PATH/train.json:$TRAIN_DATASETS
export DEV_PATH=$DATA_BASE_PATH/dev.json:pwc
export TEST_PATH=$DATA_BASE_PATH/test.json:pwc

export OUTPUT_BASE_PATH=${OUTPUT_DIR:-outputs/pwc_outputs/experiment_dygie_crf_n_ary/$1}

python -m allennlp.run evaluate --output-file $OUTPUT_BASE_PATH/metrics_test.json --include-package dygie \
$OUTPUT_BASE_PATH/model.tar.gz $TEST_PATH