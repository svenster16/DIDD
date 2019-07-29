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
from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.data_generators import lm1b
from . import problems
from . import reddit
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from google.cloud import storage

import tensorflow as tf


@registry.register_problem
class Lm1bSocialMediaDepression(multi_problem_v2.MultiProblemV2):
  """LM1b and Depression mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    problemos = [lm1b.LanguagemodelLm1b32k(),
                problems.TwitterDepression(),
                 reddit.RedditDepression()]
    schedule = 'step @0 1.0 0.0 0.0 @10000 1.0 0.0 0.0'

    super(Lm1bSocialMediaDepression, self).__init__(
        problemos, schedule, was_reversed=was_reversed, was_copy=was_copy)

  @property
  def use_vocab_from_other_problem(self):
    return lm1b.LanguagemodelLm1b32k()

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

