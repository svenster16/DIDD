#!/bin/bash

PROBLEM=twitter_depression
MODEL=transformer_encoder
HPARAMS=transformer_tpu_td

BUCKET_DIR=gs://sventestbucket
USR_DIR=/home/sven/Projects/DIDD/problems/
DATA_DIR=$BUCKET_DIR/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$BUCKET_DIR/t2t_train/$PROBLEM/$MODEL-$HPARAMS
TEST_DIR=/home/sven/Desktop/Research/REU_2019/code/clpsych/data/lm-test-data/test_text.txt
TEST_RESULT_DIR=$BUCKET_DIR/test_results

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --output_dir=$TRAIN_DIR \
  --hparams_set=$HPARAMS \
  --decode_hparams="batch_size=512,beam_size=1" \
  --decode_to_file=$TEST_RESULT_DIR/test_output.txt \
  --decode_from_file=$TEST_DIR

