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
from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import lm1b
from . import problems
from . import reddit
from tensor2tensor.models import transformer
from tensor2tensor.models import evolved_transformer
from tensor2tensor.utils import registry
from google.cloud import storage

import tensorflow as tf


@registry.register_problem
class Lm1bSocialMediaDepression(multi_problem.MultiProblem):
  """LM1b and Depression mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(Lm1bSocialMediaDepression, self).__init__(was_reversed, was_copy)
    self.task_list.append(problems.LanguagemodelLm1b32kmulti())
    self.task_list.append(problems.TwitterDepression())
    self.task_list.append(reddit.RedditDepression())

  @property
  def use_vocab_from_other_problem(self):
    return lm1b.LanguagemodelLm1b32k()

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

@registry.register_problem
class Lm1bSocialMediaDepressionV2(multi_problem_v2.MultiProblemV2):
  """LM1b and Depression mixed problem class for multitask learning."""

  def __init__(self, was_reversed=False, was_copy=False):
    problemos = []
    rates = []
    for rate, also_reverse, cls in self.problems_and_rates:
        for r in [False, True] if also_reverse else [False]:
            problemos.append(cls(was_reversed=r))
            rates.append(rate)
    pmf = multi_problem_v2.epoch_rates_to_pmf(problemos, epoch_rates=rates)
    schedule = multi_problem_v2.constant_schedule(pmf)
    super(Lm1bSocialMediaDepression, self).__init__(
        problemos, schedule, was_reversed=was_reversed, was_copy=was_copy)
  @property
  def problems_and_rates(self):
    """Returns a list of (weight, also_reverse, problem_class) triples."""
    return [
        (1.0, False, problems.LanguagemodelLm1b32kmulti),
        (50.0, False, problems.TwitterDepression),
        (10.0, False, reddit.RedditDepression),
    ]
  @property
  def use_vocab_from_other_problem(self):
    return lm1b.LanguagemodelLm1b32k()

  @property
  def vocab_type(self):
    return text_problems.VocabType.SUBWORD

@registry.register_hparams
def transformer_tall_finetune_textclass_didd():
  """Hparams for transformer on LM for finetuning on text class problems."""
  hparams = transformer.transformer_tall_big()
  hparams.learning_rate_constant = 6.25e-5
  hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
  hparams.multiproblem_schedule_max_examples = 0
  hparams.multiproblem_target_eval_only = True
  hparams.learning_rate_warmup_steps = 50
  # Set train steps to learning_rate_decay_steps or less
  hparams.learning_rate_decay_steps = 25000
  hparams.multiproblem_reweight_label_loss = True
  hparams.multiproblem_label_weight = 0.95
  return hparams


@registry.register_hparams
def evolved_transformer_big_didd():
  """Big parameters for Evolved Transformer model on TPU."""
  hparams = evolved_transformer.add_evolved_transformer_hparams(transformer_tall_finetune_textclass_didd())

  return hparams
@registry.register_hparams
def evolved_transformer_big_tpu_didd():
  """Big parameters for Evolved Transformer model on TPU."""
  hparams = evolved_transformer_big_didd()
  transformer.update_hparams_for_tpu(hparams)
  hparams.max_length = 1024
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 32768  # max fitting in 16G memory is 49152, batch 2
  hparams.batch_size = 4
  hparams.multiproblem_vocab_size = 2**15
  return hparams
