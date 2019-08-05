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
    bucket = storage_client.get_bucket('svenbucky')

    zip_filename = 'reddit_data.zip'
    zip_filepath = os.path.join(tmp_dir, zip_filename)
    zip_blob = bucket.blob(zip_filename)

    if not os.path.exists(zip_filepath):
        zip_blob.download_to_filename(zip_filepath)
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

@registry.register_problem
class RedditDepression(text_problems.Text2ClassProblem):
    """Twitter depression classification."""

    @property
    def max_subtoken_length(self):
        return 18
    @property
    def dataset_splits(self):
        #Splits of data to produce and number of output shards for each.
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 20,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 20,
        }]
    @property
    def is_generate_per_split(self):
        return True
    @property
    def approx_vocab_size(self):
        return 2 ** 15
    @property
    def num_classes(self):
        return 2
    def class_labels(self, data_dir):
        del data_dir
        return ["Control", "Depression"]
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "dev"
        dirs = [(os.path.join(tmp_dir, 'RSDD', dataset, "depression"), True), (os.path.join(
            tmp_dir, dataset, "control"), False)]
        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as f:
                    for line in f:
                        yield {
                            "inputs": line,
                            "label": int(label),
                        }

    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ROC_AUC
        ]
@registry.register_problem
class RedditDepressionCharacters(RedditDepression):
  """Reddit depresssion classification, character level."""

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
