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
                        "input": twt.text.strip(),
                        "label": int(female),
                    }
            except ET.ParseError:
                pass
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