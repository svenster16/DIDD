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
from tensor2tensor.models import text_cnn
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
import xml.etree.ElementTree as ET
import tensorflow as tf

def download_blob(tmp_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('svenbucky')

    zip_filename = 'PAN.zip'
    zip_filepath = os.path.join(tmp_dir,zip_filename)
    zip_blob = bucket.blob(zip_filename)

    if not os.path.exists(zip_filepath):
        zip_blob.download_to_filename(zip_filepath)
        import zipfile
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)

@registry.register_problem
class AgePAN(text_problems.Text2ClassProblem):
    """Twitter depression classification."""
    @property
    def max_subtoken_length(self):
        return 18

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        users = {}
        dataset = "train" if dataset_split == problem.DatasetSplit.TRAIN else "test"
        with open(os.path.join(tmp_dir, dataset, 'truth.txt'), 'r') as fout:
            for line in fout:
                line = line.strip().split(':::')
                users[line[0]] = {"gender": line[1], "age": line[2], "extroverted": line[3], "stable": line[4],
                                  "agreeable": line[5], "conscientious": line[6], "open": line[7]}
        for file in os.listdir(os.path.join(tmp_dir, dataset)):
            userid = os.path.splitext(file)
            try:
                root = ET.parse(os.path.join(tmp_dir, dataset, file))
                female = True if users[userid[0]]["gender"].strip() == "F" else False
                for twt in root.iter('document'):
                    yield {
                        "inputs": twt.text.strip(),
                        "label": int(female),
                    }
            except ET.ParseError:
                pass
    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 3,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    @property
    def is_generate_per_split(self):
        return False
    @property
    def approx_vocab_size(self):
        return 2 ** 13  # 32k
    @property
    def num_classes(self):
        return 2
    def eval_metrics(self):
        return [
            metrics.Metrics.ACC, metrics.Metrics.ROC_AUC
        ]
    def class_labels(self, data_dir):
        del data_dir
        return ["male", "female"]

@registry.register_problem
class AgeAggPAN(AgePAN):
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        users = {}
        dataset = "train" if dataset_split == problem.DatasetSplit.TRAIN else "test"
        with open(os.path.join(tmp_dir, dataset, 'truth.txt'), 'r') as fout:
            for line in fout:
                line = line.strip().split(':::')
                users[line[0]] = {"gender": line[1], "age": line[2], "extroverted": line[3], "stable": line[4],
                                  "agreeable": line[5], "conscientious": line[6], "open": line[7]}
        for file in os.listdir(os.path.join(tmp_dir, dataset)):
            userid = os.path.splitext(file)

            try:
                aggtext = ''
                root = ET.parse(os.path.join(tmp_dir, dataset, file))
                female = True if users[userid[0]]["gender"].strip() == "F" else False
                for twt in root.iter('document'):
                    aggtext = aggtext + ' <EOP> ' + twt.text.strip()
                yield {
                    "inputs": aggtext,
                    "label": int(female),
                }
            except ET.ParseError:
                pass

@registry.register_problem
class AgeTwitter(AgePAN):
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        download_blob(tmp_dir)
        users = {}
        dataset = "train" if dataset_split == problem.DatasetSplit.TRAIN else "test"
        with open(os.path.join(tmp_dir, dataset, 'truth.txt'), 'r') as fout:
            for line in fout:
                line = line.strip().split(':::')
                users[line[0]] = {"gender": line[1], "age": line[2], "extroverted": line[3], "stable": line[4],
                                  "agreeable": line[5], "conscientious": line[6], "open": line[7]}
        for file in os.listdir(os.path.join(tmp_dir, dataset)):
            userid = os.path.splitext(file)
            try:
                root = ET.parse(os.path.join(tmp_dir, dataset, file))
                age = users[userid[0]]["age"].strip()
                if age == "18-24":
                    label = 0
                elif age == "25-34":
                    label = 1
                elif age == "35-49":
                    label = 2
                else:
                    label = 3
                for twt in root.iter('document'):
                    yield {
                        "inputs": twt.text.strip(),
                        "label": label,
                    }
            except ET.ParseError:
                pass

    @property
    def num_classes(self):
        return 4
    def class_labels(self, data_dir):
        del data_dir
        return ["18-24", "25-34", "35-49", "50-XX"]
    def eval_metrics(self):
        return [
            metrics.Metrics.ACC
        ]

@registry.register_hparams
def text_cnn_tiny():
    hparams = text_cnn.text_cnn_base()
    hparams.batch_size = 1028
    hparams.num_hidden_layers = 1
    hparams.num_filters = 64
    hparams.ouput_dropout = 0.7
    return hparams

@registry.register_hparams
def transformer_extra_tiny():
    hparams = transformer.transformer_tiny()
    hparams.num_hidden_layers = 1
    hparams.hidden_size = 32
    hparams.filter_size = 128
    hparams.num_heads = 2
    return hparams
@registry.register_hparams
def transformer_extra_tiny_agg():
    hparams = transformer.transformer_tiny()
    hparams.batch_size = 8
    hparams.num_hidden_layers = 1
    hparams.hidden_size = 64
    hparams.filter_size = 256
    hparams.num_heads = 2
    return hparams

@registry.register_hparams
def transformer_extra_tiny_textclass():
    hparams = transformer_extra_tiny()
    hparams.layer_prepostprocess_dropout = 0.1
    hparams.learning_rate_warmup_steps = 2000
    hparams.learning_rate_constant = 4e-7
    hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
    # Set train steps to learning_rate_decay_steps or less
    hparams.learning_rate_decay_steps = 10000
    return hparams