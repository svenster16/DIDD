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
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from google.cloud import storage

import tensorflow as tf

def _original_vocab(tmp_dir):
  """Returns a set containing the original vocabulary.
  This is important for comparing with published results.
  Args:
    tmp_dir: directory containing dataset.
  Returns:
    a set of strings
  """
  vocab_url = ("http://download.tensorflow.org/models/LM_LSTM_CNN/"
               "vocab-2016-09-10.txt")
  vocab_filename = os.path.basename(vocab_url + ".en")
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if not os.path.exists(vocab_filepath):
    generator_utils.maybe_download(tmp_dir, vocab_filename, vocab_url)
  return set([
      text_encoder.native_to_unicode(l.strip())
      for l in tf.gfile.Open(vocab_filepath)
  ])

def _replace_oov(original_vocab, line):
  """Replace out-of-vocab words with "UNK".
  This maintains compatibility with published results.
  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.
  Returns:
    a unicode string - a space-delimited sequence of words.
  """
  return u" ".join(
      [word if word in original_vocab else u"UNK" for word in line.split()])

def download_blob(tmp_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('sventestbucket')
    corpus_filename = 'reddit_data.zip'
    test_filename = 'reddit_test_set.txt'
    test_filepath = os.path.join(tmp_dir,test_filename)
    vocab_filename = 'vocab.depression_twitter.32768.subwords'
    corpus_filepath = os.path.join(tmp_dir,corpus_filename)
    vocab_filepath = os.path.join(tmp_dir,vocab_filename)
    blob = bucket.blob('reddit_data/'+corpus_filename)
    testdatablob = bucket.blob('reddit_data/' + corpus_filename)
    blob2 = bucket.blob(vocab_filename)
    if not os.path.exists(vocab_filepath):
        blob2.download_to_filename(vocab_filepath)
    if not os.path.exists(test_filepath):
        testdatablob.download_to_filename(test_filepath)
    if not os.path.exists(corpus_filepath):
        blob.download_to_filename(corpus_filepath)
        import zipfile
        with zipfile.ZipFile(corpus_filepath, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

def _train_data_filenames(tmp_dir):
    return [
      (os.path.join(tmp_dir,
                   "control_training.txt"), False),
      (os.path.join(tmp_dir,
                   "depression_training.txt"), True)
    ]

def _dev_data_filenames(tmp_dir):
    return [
      (os.path.join(tmp_dir,
                   "control_validation.txt"), False),
      (os.path.join(tmp_dir,
                   "depression_validation.txt"), True)
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
        }]
    """
    @property
    def dataset_splits(self):
        Splits of data to produce and number of output shards for each.
        return [{
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
       }]

    @property
    def already_shuffled(self):
        return True
    """
    @property
    def is_generate_per_split(self):
        return True
    @property
    def approx_vocab_size(self):
        return 2 ** 15  # 32768
    @property
    def num_classes(self):
        return 2

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
        original_vocab = _original_vocab(tmp_dir)
        files = split_files[dataset_split]
        for filepath, label in files:
            tf.logging.info("filepath = %s", filepath)
            for line in tf.gfile.Open(filepath):
                txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
                yield {
                    "inputs": txt,
                    "label": int(label),
                }
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

@registry.register_hparams
def transformer_oom():
  """
  HParams for Transformer model on TPU and
  finetuned for twitter depression (td) classification.
  """
  hparams = transformer.transformer_base()
  hparams.batch_size = 2048
  return hparams
