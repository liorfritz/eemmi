# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================

# pylint: disable=unused-import
"""HMM (hidden Markov model) MMI (maximum mutual information) Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.nn_grad import _BroadcastMul
import numpy as np

gen_hmm_mmi_ops = tf.load_op_library('gen_hmm_mmi_ops.so')

def hmm_mmi_loss(inputs, labels, sequence_length, priors, transition_probs, lang_transition_probs,
             preprocess_collapse_repeated=False, ctc_merge_repeated=True):
  """Computes the CTC (Connectionist Temporal Classification) Loss.

  Requires:
    ```sequence_length(b) <= time for all b

    max(labels.indices(labels.indices[:, 1] == b, 2))
      <= sequence_length(b) for all b.```

  If ctc_merge_repeated is set False, then *during* CTC calculation
  repeated non-blank labels will not be merged and are interpreted
  as individual labels.  This is a simplified version of CTC.


  Args:
    inputs: 3-D `float` `Tensor` sized
      `[max_time x batch_size x num_classes]`.  The logits.
    labels: An `int32` `SparseTensor`.
      `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
      the id for (batch b, time t).  See `core/ops/ctc_ops.cc` for more details.
    sequence_length: 1-D `int32` vector, size `[batch_size]`.
      The sequence lengths.
    priors: 1-D `float` `Tensor`, size `[num_classes]`.
      The prior probabilities of observing each of the classes.
    transition_probs: 2-D `float` `Tensor` sized `[num_classes x 3]`.
      The transition probabilities between classes (only three transitions are possible):
      transition_probs[i, 0] = probability of self transition of class i.
      transition_probs[i, 1] = probability of transition from class i to blank label (class: num_classes - 1)
      transition_probs[i, 2] = probability of transition from class i to a non blank label.
    preprocess_collapse_repeated: Boolean.  Default: False.
      If True, repeated labels are collapsed prior to the CTC calculation.
    ctc_merge_repeated: Boolean.  Default: True.


  Returns:
    A 1-D `float` `Tensor`, size `[batch]`, containing logits.


  Raises:
    TypeError: if labels is not a `SparseTensor`.
  """
  # The second, third, etc output tensors contain the gradients.  We use it in
  # _CTCLossGrad() below.
  if not isinstance(labels, sparse_tensor.SparseTensor):
    raise TypeError("Expected labels to be a SparseTensor")
  loss, _, _, _ = gen_hmm_mmi_ops.hmm_mmi_loss(
      inputs,
      labels.indices,
      labels.values,
      sequence_length,
      priors,
      transition_probs,
      lang_transition_probs,
      preprocess_collapse_repeated=preprocess_collapse_repeated,
      hmm_merge_repeated=ctc_merge_repeated)

  return loss


# pylint: disable=unused-argument
@ops.RegisterGradient("HmmMmiLoss")
def _HmmMmiLossGrad(op, grad_loss, *unused):
  """The derivative provided by CTC Loss.

  Args:
     op: the CTCLoss op.
     grad_loss: The backprop for cost.

#   Returns:
#      The CTC Loss gradient.
#   """
  # Outputs are: loss, grad
  grad = op.outputs[1]
  priors_grad = op.outputs[2]
  trans_grad = op.outputs[3]
  # Return gradient for inputs and None for
  # labels_indices, labels_values and sequence_length
  return [_BroadcastMul(grad_loss, grad), None, None, None, priors_grad, trans_grad, None]

