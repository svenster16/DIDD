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
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.data_generators import lm1b
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators import imdb
from random import randrange
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from google.cloud import storage

import tensorflow as tf

def download_blob(tmp_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('sventestbucket')

    zip_filename = 'twitter_final_depression_data.zip'
    zip_filepath = os.path.join(tmp_dir,zip_filename)
    zip_blob = bucket.blob('data/' + zip_filename)

    if not os.path.exists(zip_filepath):
        zip_blob.download_to_filename(zip_filepath)
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

@registry.register_problem
class TwitterDepression(text_problems.Text2ClassProblem):
    """Twitter depression classification."""
    @property
    def max_subtoken_length(self):
        return 18
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        # Generate examples
        #original_vocab = _original_vocab(tmp_dir)
        # txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
        train = dataset_split == problem.DatasetSplit.TRAIN
        dev = dataset_split == problem.DatasetSplit.EVAL
        if train:
            dataset = "train"
        elif dev:
            dataset = "dev"
        else:
            dataset = "test"
        dirs = [(os.path.join(tmp_dir,"twitter_depression_data", dataset, "depression"), True), (os.path.join(
            tmp_dir,"twitter_depression_data",dataset, "control"), False)]

        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as f:
                    for line in f:
                        num = randrange(3)
                        yield {
                            "inputs": line,
                            "label": int(label),
                        }
                        """over sampling depression postsquit"""
                        if (num == 0 or num == 1) and label and dataset != "test":
                            yield {
                                "inputs": line,
                                "label": int(label),
                            }
    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    @property
    def is_generate_per_split(self):
        return True
    @property
    def approx_vocab_size(self):
        return 2 ** 15  # 32k
    @property
    def num_classes(self):
        return 2
    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ROC_AUC
        ]
    def class_labels(self, data_dir):
        del data_dir
        return ["Control", "Depression"]

@registry.register_problem
class TwitterDepressionVanilla(TwitterDepression):

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        download_blob(tmp_dir)
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "dev"
        dirs = [(os.path.join(tmp_dir,"twitter_final_depression_data", dataset, "depression"), True), (os.path.join(
            tmp_dir,"twitter_final_depression_data",dataset, "control"), False)]
        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as f:
                    for line in f:
                        yield {
                            "inputs": line,
                            "label": int(label),
                        }
    """

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        download_blob(tmp_dir)
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "test"
        filepath = os.path.join(tmp_dir, "twitter_depression_data", dataset, 'test_text.txt')
        with tf.gfile.Open(filepath) as f:
            for line in f:
                yield {
                    "inputs": line,
                    "label": 0,
                }
    """
    """
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    @property
    def already_shuffled(self):
        return True
    """

@registry.register_problem
class TwitterDepressionAgg20Vanilla(TwitterDepression):
    @property
    def aggragate_number(self):
        return 20
    """
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        download_blob(tmp_dir)
        # Generate examples
        # original_vocab = _original_vocab(tmp_dir)
        # txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "dev"
        dirs = [(os.path.join(tmp_dir, "twitter_depression_data", dataset, "depression"), True), (os.path.join(
            tmp_dir, "twitter_depression_data", dataset, "control"), False)]
        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as f:
                    count = 0
                    txt = ''
                    for line in f:
                        if count == self.aggragate_number:
                            yield {
                                "inputs": txt,
                                "label": int(label),
                            }
                            count = 0
                            txt = ''
                            continue
                        count += 1
                        txt = txt + ' ' + line
    """
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        download_blob(tmp_dir)
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "test_agg"
        filepath = os.path.join(tmp_dir, "twitter_final_depression_data", dataset, 'test_text_agg.txt')
        with tf.gfile.Open(filepath) as f:
            count = 0
            agg = ''
            for line in f:
                if count == self.aggragate_number:
                    yield {
                        "inputs": agg,
                        "label": 0,
                    }
                    count = 0
                    agg = ''
                    continue
                count += 1
                agg = agg + ' ' + line

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    @property
    def already_shuffled(self):
        return True
@registry.register_problem
class TwitterDepressionCharactersAgg20(TwitterDepressionAgg20Vanilla):
    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def global_task_id(self):
        return problem.TaskID.EN_CHR_SENT
@registry.register_problem
class LanguagemodelLm1b32kmulti(lm1b.LanguagemodelLm1b32k):
    @property
    def num_training_examples(self):
        return 30301028
@registry.register_problem
class TwitterDepressionCharacters(TwitterDepression):
  """Twitter depresssion classification, character level."""
  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT

@registry.register_problem
class MultiTwitterWikiLMSharedVocab64k(TwitterDepression):
  """MultiNLI classification problems with the Wiki vocabulary."""
  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki64k()

@registry.register_hparams
def transformer_tall_tpu():
  """
  HParams for Transformer model on TPU and 
  finetuned for twitter depression (td) classification.
  """
  hparams = transformer.transformer_tall_finetune_textclass()
  transformer.update_hparams_for_tpu(hparams)
  return hparams

@registry.register_hparams
def transformer_textclass_big():
  hparams = transformer.transformer_big()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 50
  hparams.learning_rate_constant = 6.25e-6
  hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 20000
  return hparams

@registry.register_hparams
def transformer_textclass_base():
  hparams = transformer.transformer_base()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 50
  hparams.learning_rate_constant = 6.25e-6
  hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 20000
  return hparams

@registry.register_problem
class SentimentIMDBTest(imdb.SentimentIMDB):

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]