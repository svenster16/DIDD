# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Edited by Sven Marnauzs for academic research use

"""Twitter Depression Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from six.moves import range
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import lm1b
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from google.cloud import storage

import tensorflow as tf

def download_blob(tmp_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('sventestbucket')

    zip_filename = 'reddit_data_shuf.zip'
    zip_filepath = os.path.join(tmp_dir, zip_filename)
    zip_blob = bucket.blob('reddit_didd_data/' + zip_filename)

    test_filename = 'reddit_test_set.txt'
    test_filepath = os.path.join(tmp_dir,test_filename)
    test_blob = bucket.blob('reddit_didd_data/' + zip_filename)

    if not os.path.exists(test_filepath):
        test_blob.download_to_filename(test_filepath)
    if not os.path.exists(zip_filepath):
        zip_blob.download_to_filename(zip_filepath)
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

def _train_data_filenames(tmp_dir):
    return [
      (os.path.join(tmp_dir,
                   "control_training_shuf.txt"), False),
      (os.path.join(tmp_dir,
                   "depression_training_shuf.txt"), True)
    ]

def _dev_data_filenames(tmp_dir):
    return [
      (os.path.join(tmp_dir,
                   "control_validation_shuf.txt"), False),
      (os.path.join(tmp_dir,
                   "depression_validation_shuf.txt"), True)
    ]
def _test_data_filenames(tmp_dir):
    return [
      (os.path.join(tmp_dir,
                   "reddit_test_set.txt"), False)
      ]

@registry.register_problem
class RedditDepression(text_problems.Text2ClassProblem):
    """Twitter depression classification."""
    @property
    def dataset_splits(self):
        #Splits of data to produce and number of output shards for each.
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 50,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 50,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
       }]

    @property
    def already_shuffled(self):
        return True
    @property
    def is_generate_per_split(self):
        return True
    #@property
    #def vocab_filename(self):
        #return lm1b.LanguagemodelLm1b32k().vocab_filename
    @property
    def approx_vocab_size(self):
        return 2 ** 16
    @property
    def num_classes(self):
        return 2
    """
    @property
    def num_training_examples(self):
            #Used when mixing problems - how many examples are in the dataset
        return 10958022
    """

    def class_labels(self, data_dir):
        del data_dir
        return ["Control", "Depression"]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        # Generate examples
        split_files = {
            problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
            problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
            problem.DatasetSplit.TEST: _test_data_filenames(tmp_dir),
        }
        files = split_files[dataset_split]
        for filepath, label in files:
            tf.logging.info("filepath = %s", filepath)
            for line in tf.gfile.Open(filepath):
                yield {
                    "inputs": line,
                    "label": int(label),
                }

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ROC_AUC, metrics.Metrics.SET_PRECISION, metrics.Metrics.SET_RECALL
        ]
@registry.register_problem
class RedditDepressionCharacters(RedditDepression):
  """Reddit depresssion classification, character level."""

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
      """Generate examples."""
      download_blob(tmp_dir)
      # Generate examples
      split_files = {
          problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
          problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
      }
      files = split_files[dataset_split]
      for filepath, label in files:
          tf.logging.info("filepath = %s", filepath)
          for line in tf.gfile.Open(filepath):
              yield {
                    "inputs": line,
                    "label": int(label),
                }
  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT

@registry.register_problem
class RedditDepressionAgg10(RedditDepression):
    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        POST_AGG_COUNT = 10 #number of posts to aggragate together

        download_blob(tmp_dir)
        # Generate examples
        split_files = {
            problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
            problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
            problem.DatasetSplit.TEST: _test_data_filenames(tmp_dir),
        }
        files = split_files[dataset_split]
        for filepath, label in files:
            tf.logging.info("filepath = %s", filepath)
            count = 0
            aggtext = ''
            for line in tf.gfile.Open(filepath):
                if count == POST_AGG_COUNT:
                    yield {
                        "inputs": aggtext,
                        "label": int(label),
                    }
                    count = 0
                    aggtext = ''
                    continue
                aggtext = aggtext + ' ' + line
                count += 1

@registry.register_problem
class RedditDepressionAgg20(RedditDepression):
    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        POST_AGG_COUNT = 20 #number of posts to aggragate together

        download_blob(tmp_dir)
        # Generate examples
        split_files = {
            problem.DatasetSplit.TRAIN: _train_data_filenames(tmp_dir),
            problem.DatasetSplit.EVAL: _dev_data_filenames(tmp_dir),
            problem.DatasetSplit.TEST: _test_data_filenames(tmp_dir),
        }
        files = split_files[dataset_split]
        for filepath, label in files:
            tf.logging.info("filepath = %s", filepath)
            count = 0
            aggtext = ''
            for line in tf.gfile.Open(filepath):
                if count == POST_AGG_COUNT:
                    yield {
                        "inputs": aggtext,
                        "label": int(label),
                    }
                    count = 0
                    aggtext = ''
                    continue
                aggtext = aggtext + ' ' + line
                count += 1
@registry.register_hparams
def transformer_oom():
  """
  HParams for Transformer model on TPU and
  finetuned for twitter depression (td) classification.
  """
  hparams = transformer.transformer_base()
  hparams.batch_size = 2048
  return hparams
