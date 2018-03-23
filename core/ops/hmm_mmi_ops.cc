/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "../util/hmm/hmm_mmi_loss_calculator.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
const double kLogZero = -std::numeric_limits<double>::infinity();

// MMI is maximum mutual information

REGISTER_OP("HmmMmiLoss")
    .Attr("T: numbertype")
    .Input("inputs: float")
    .Input("labels_indices: int64")
    .Input("labels_values: int32")
    .Input("sequence_length: int32")
    .Input("priors: T")
    .Input("transition_probs: T")
    .Input("lang_transition_probs: T")
    .Attr("preprocess_collapse_repeated: bool = false")
    .Attr("hmm_merge_repeated: bool = true")
    .Output("loss: float")
    .Output("gradient: float")
    .Output("priors_gradient: float")
    .Output("trans_gradient: float")
    .Doc(R"doc(
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle labels_indices;
      ShapeHandle labels_values;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &labels_indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &labels_values));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sequence_length));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(labels_indices, 0),
                                  c->Dim(labels_values, 0), &unused));

      // Get batch size from inputs and sequence_length, and update inputs
      // with the merged batch_size since it is returned.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));
      TF_RETURN_IF_ERROR(c->ReplaceDim(inputs, 1, batch_size, &inputs));

      c->set_output(0, c->Vector(batch_size));
      c->set_output(1, inputs);
      return Status::OK();
    })
    .Doc(R"doc(
Calculates the MMI Loss (log probability) for each batch entry.  Also calculates
the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
labels_indices: The indices of a `SparseTensor<int32, 2>`.
  `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
  `(batch b, time t)`.
labels_values: The values (labels) associated with the given batch and time.
sequence_length: A vector containing sequence lengths (batch).
loss: A vector (batch) containing log-probabilities.
gradient: The gradient of `loss`.  3-D, shape:
  `(max_time x batch_size x num_classes)`.
)doc");

class HmmMmiLossOp : public OpKernel {
  typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor> >
      InputMap;
  typedef Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
      OutputMap;

 public:
  explicit HmmMmiLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated",
                                     &preprocess_collapse_repeated_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("hmm_merge_repeated", &hmm_merge_repeated_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* labels_indices;
    const Tensor* labels_values;
    const Tensor* seq_len;
    const Tensor* priors;
    const Tensor* transition_probs;
    const Tensor* lang_transition_probs;
    OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs));
    OP_REQUIRES_OK(ctx, ctx->input("labels_indices", &labels_indices));
    OP_REQUIRES_OK(ctx, ctx->input("labels_values", &labels_values));
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));
    OP_REQUIRES_OK(ctx, ctx->input("priors", &priors));
    OP_REQUIRES_OK(ctx, ctx->input("transition_probs", &transition_probs));
    OP_REQUIRES_OK(ctx, ctx->input("lang_transition_probs", &lang_transition_probs));

    OP_REQUIRES(ctx, inputs->shape().dims() == 3,
                errors::InvalidArgument("inputs is not a 3-Tensor"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(seq_len->shape()),
                errors::InvalidArgument("sequence_length is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(labels_indices->shape()),
                errors::InvalidArgument("labels_indices is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_values->shape()),
                errors::InvalidArgument("labels_values is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(priors->shape()),
                errors::InvalidArgument("priors is not a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(transition_probs->shape()),
                errors::InvalidArgument("transition_probs is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(lang_transition_probs->shape()),
                errors::InvalidArgument("lang_transition_probs is not a matrix"));

    const TensorShape& inputs_shape = inputs->shape();
    const TensorShape& priors_shape = priors->shape();
    const TensorShape& trans_shape = transition_probs->shape();
    const TensorShape& lang_trans_shape = lang_transition_probs->shape();
    const int64 max_time = inputs_shape.dim_size(0);
    const int64 batch_size = inputs_shape.dim_size(1);
    //priors_gradient_shape.InsertDim(0, batch_size);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);
    OP_REQUIRES(
        ctx, batch_size == seq_len->dim_size(0),
        errors::InvalidArgument("len(sequence_length) != batch_size.  ",
                                "len(sequence_length):  ", seq_len->dim_size(0),
                                " batch_size: ", batch_size));

    auto seq_len_t = seq_len->vec<int32>();
    auto priors_t = priors->vec<float>();
    auto transition_probs_t = transition_probs->matrix<float>();
    auto lang_transition_probs_t = lang_transition_probs->matrix<float>();

    OP_REQUIRES(ctx, labels_indices->dim_size(0) == labels_values->dim_size(0),
                errors::InvalidArgument(
                    "labels_indices and labels_values must contain the "
                    "same number of rows, but saw shapes: ",
                    labels_indices->shape().DebugString(), " vs. ",
                    labels_values->shape().DebugString()));

    TensorShape labels_shape({batch_size, max_time});
    std::vector<int64> order{0, 1};
    sparse::SparseTensor labels_sp(*labels_indices, *labels_values,
                                   labels_shape, order);

    Status labels_sp_valid = labels_sp.IndicesValid();
    OP_REQUIRES(ctx, labels_sp_valid.ok(),
                errors::InvalidArgument("label SparseTensor is not valid: ",
                                        labels_sp_valid.error_message()));

    hmm::HmmMmiLossCalculator::LabelSequences labels_t(batch_size);
    for (const auto& g : labels_sp.group({0})) {  // iterate by batch
      const int64 batch_indices = g.group()[0];
      OP_REQUIRES(ctx, FastBoundsCheck(batch_indices, batch_size),
                  errors::InvalidArgument("labels batch index must be between ",
                                          0, " and ", batch_size, " but saw: ",
                                          batch_indices));

      auto values = g.values<int32>();
      std::vector<int>* b_values = &labels_t[batch_indices];
      b_values->resize(values.size());
      for (int i = 0; i < values.size(); ++i) (*b_values)[i] = values(i);
    }

    OP_REQUIRES(ctx, static_cast<size_t>(batch_size) == labels_t.size(),
                errors::InvalidArgument("len(labels) != batch_size.  ",
                                        "len(labels):  ", labels_t.size(),
                                        " batch_size: ", batch_size));

    for (int64 b = 0; b < batch_size; ++b) {
      OP_REQUIRES(
          ctx, seq_len_t(b) <= max_time,
          errors::InvalidArgument("sequence_length(", b, ") <= ", max_time));
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len->shape(), &loss));
    auto loss_t = loss->vec<float>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<float, 3>();
    auto inputs_t = inputs->tensor<float, 3>();
    std::vector<OutputMap> gradient_list_t;
    std::vector<InputMap> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
      gradient_list_t.emplace_back(
          gradient_t.data() + t * batch_size * num_classes, batch_size,
          num_classes);
    }

    gradient_t.setZero();

    Tensor* priors_gradient;
    OP_REQUIRES_OK(ctx,
               ctx->allocate_output("priors_gradient", priors_shape, &priors_gradient));
    auto priors_gradient_t = priors_gradient->vec<float>();
    priors_gradient_t.setZero();

    Tensor* trans_gradient;
    OP_REQUIRES_OK(ctx,
               ctx->allocate_output("trans_gradient", trans_shape, &trans_gradient));

    auto trans_gradient_t = trans_gradient->matrix<float>();
    trans_gradient_t.setZero();

    // Assumption: the blank index is num_classes - 1
    hmm::HmmMmiLossCalculator hmm_mmi_loss_calculator(num_classes - 1, 0);
    DeviceBase::CpuWorkerThreads workers =
        *ctx->device()->tensorflow_cpu_worker_threads();
    OP_REQUIRES_OK(ctx, hmm_mmi_loss_calculator.CalculateLoss(
                            seq_len_t, labels_t, input_list_t, priors_t, transition_probs_t, lang_transition_probs_t,
                            preprocess_collapse_repeated_, hmm_merge_repeated_,
                            &loss_t, &gradient_list_t, &priors_gradient_t, &trans_gradient_t, &workers));
  }

 private:
  bool preprocess_collapse_repeated_;
  bool hmm_merge_repeated_;
  TF_DISALLOW_COPY_AND_ASSIGN(HmmMmiLossOp);
};

REGISTER_KERNEL_BUILDER(Name("HmmMmiLoss").Device(DEVICE_CPU), HmmMmiLossOp);

}  // namespace tensorflow
