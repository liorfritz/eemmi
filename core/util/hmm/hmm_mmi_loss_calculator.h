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

#ifndef TENSORFLOW_CORE_UTIL_HMM_MMI_LOSS_CALCULATOR_H_
#define TENSORFLOW_CORE_UTIL_HMM_MMI_LOSS_CALCULATOR_H_

#include <vector>
#include <iomanip>
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/work_sharder.h"
#include <ctime>

int T_maxes = 3; // time frames period, after which we subtract the max coeff of alpha/beta from alpha/beta (for numerical stability)
const double kLogZero = -std::numeric_limits<double>::infinity();

namespace tensorflow {
namespace hmm {

using strings::StrCat;

class HmmMmiLossCalculator {
  // Hidden Markov Model - Maximum Mutual Information Loss
  //
  // Implementation by lior.fritz@gmail.com
  //
  // The HMM-MMI loss layer learns the posteriors, priors and transition probabilities defined by the HMM model.
  //
  // Reference materials: Simplified End-to-End MMI Training and Voting for ASR (https://arxiv.org/abs/1703.10356)
  //  .
 public:
  typedef std::vector<std::vector<int>> LabelSequences;
  typedef Eigen::MatrixXd Matrix;
  typedef Eigen::ArrayXd Array;
  typedef Eigen::Map<const Eigen::MatrixXf> InputMap;
  typedef Eigen::Map<Eigen::MatrixXf> OutputMap;

  HmmMmiLossCalculator(int blank_index, int output_delay)
      : blank_index_(blank_index), output_delay_(output_delay) {}

  template <typename VectorIn, typename VectorOut, typename MatrixIn,
            typename MatrixOut, typename VectorInFloat, typename MatrixInFloat, typename VectorOutPriors, typename MatrixOutTrans>
  Status CalculateLoss(const VectorIn& seq_len, const LabelSequences& labels,
                       const std::vector<MatrixIn>& inputs, VectorInFloat& priors, MatrixInFloat& transition_probs, MatrixInFloat& lang_transition_probs,
                       bool preprocess_collapse_repeated,
                       bool hmm_merge_repeated, VectorOut* loss,
                       std::vector<MatrixOut>* gradients, VectorOutPriors* priors_gradient, MatrixOutTrans* trans_gradient,
                       DeviceBase::CpuWorkerThreads* workers = nullptr) const;

private:

  // Numerator forward-backward
  void CalculateAlpha(const std::vector<int>& l_prime,
                                 const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, const Array& priors, bool hmm_merge_repeated,
                                 Matrix* log_alpha, Array* log_alpha_maxes) const;

  void CalculateBeta(const std::vector<int>& l_prime,
                                  const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, bool hmm_merge_repeated,
                                  Matrix* log_beta, Array* log_beta_maxes) const;

  // Denominator forward-backward
  void CalculateTildeAlpha(const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, const Array& priors, bool hmm_merge_repeated,
                                 Matrix* log_alpha, Array* log_alpha_maxes) const;

  void CalculateTildeBeta(const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, bool hmm_merge_repeated,
                                  Matrix* log_beta, Array* log_beta_maxes) const;

  // Gradients
  void CalculateGradientsNumerator(const std::vector<int>& l_prime, const Matrix& y,
                         Matrix& log_alpha, const Array& log_alpha_maxes, Matrix& log_beta, const Array& log_beta_maxes,
                         const Array& log_p_z_x, const Array& priors, const Matrix& t_p, const Matrix& l_t_p,
                         Eigen::MatrixXf* dy, Eigen::ArrayXf* d_priors, Eigen::MatrixXf* d_t_p, Array* alpha_beta_over_t_num,
                         double log_beta_maxes_sum) const;
  void CalculateGradientsDenominator(const Matrix& y,
                         Matrix& log_alpha, const Array& log_alpha_maxes, Matrix& log_beta, const Array& log_beta_maxes,
                         const Array& log_p_z_x, const Array& priors, const Matrix& t_p, const Matrix& l_t_p, const Array& alpha_beta_over_t_num,
                         Eigen::MatrixXf* dy, Eigen::ArrayXf* d_priors, Eigen::MatrixXf* d_t_p,
                         double log_beta_maxes_sum) const;

  void GetLPrimeIndicesHMM(const std::vector<int>& l,
                        std::vector<int>* l_prime) const;

  // Helper function that calculates the l_prime indices for all
  // batches at the same time, and identifies errors for any given
  // batch.  Return value:
  //    max_{b in batch_size} l_primes[b].size()
  template <typename Vector>
  Status PopulateLPrimesHMM(bool preprocess_collapse_repeated, int batch_size,
                         int num_classes, const Vector& seq_len,
                         const LabelSequences& labels, size_t* max_u_prime,
                         LabelSequences* l_primes) const;

  // Utility indices for the algorithm.
  int blank_index_;

  // Delay for target labels in time steps.
  // The delay in time steps before the output sequence.
  const int output_delay_;
};

inline double LogSumExp(double log_prob_1, double log_prob_2) {
  // Always have 'b' be the smaller number to avoid the exponential from
  // blowing up.
  if (log_prob_1 == kLogZero && log_prob_2 == kLogZero) {
    return kLogZero;
  } else {
    return (log_prob_1 > log_prob_2)
               ? log_prob_1 + log1p(exp(log_prob_2 - log_prob_1))
               : log_prob_2 + log1p(exp(log_prob_1 - log_prob_2));
  }
}

template <typename VectorIn, typename VectorOut, typename MatrixIn,
          typename MatrixOut, typename VectorInFloat, typename MatrixInFloat, typename VectorOutPriors, typename MatrixOutTrans>
Status HmmMmiLossCalculator::CalculateLoss(
    const VectorIn& seq_len, const LabelSequences& labels,
    const std::vector<MatrixIn>& inputs, VectorInFloat& priors, MatrixInFloat& transition_probs, MatrixInFloat& lang_transition_probs, bool preprocess_collapse_repeated,
    bool hmm_merge_repeated, VectorOut* loss,
    std::vector<MatrixOut>* gradients, VectorOutPriors* priors_gradient, MatrixOutTrans* trans_gradient,
    DeviceBase::CpuWorkerThreads* workers) const {
  auto num_time_steps = inputs.size();

  if (loss == nullptr) {
    return errors::InvalidArgument("loss == nullptr");
  }

  bool requires_backprop = (gradients != nullptr);

  auto batch_size = inputs[0].rows();
  auto num_classes = inputs[0].cols();

  if (loss->size() != batch_size) {
    return errors::InvalidArgument("loss.size() != batch_size");
  }
  loss->setZero();

  for (int t = 1; t < num_time_steps; ++t) {
    if (inputs[t].rows() != batch_size) {
      return errors::InvalidArgument("Expected batch size at t: ", t,
                                     " to be: ", batch_size, " but got: ",
                                     inputs[t].rows());
    }
    if (inputs[t].cols() != num_classes) {
      return errors::InvalidArgument("Expected class count at t: ", t,
                                     " to be: ", num_classes, " but got: ",
                                     inputs[t].cols());
    }
  }

  // Check validity of sequence_length array values.
  auto max_seq_len = seq_len(0);
  for (int b = 0; b < batch_size; b++) {
    if (seq_len(b) < 0) {
      return errors::InvalidArgument("seq_len(", b, ") < 0");
    }
    if (seq_len(b) > num_time_steps) {
      return errors::InvalidArgument("seq_len(", b, ") > num_time_steps");
    }
    max_seq_len = std::max(seq_len(b), max_seq_len);
  }

  // Calculate the modified label sequence l' for each batch element,
  // and calculate the maximum necessary allocation size.
  LabelSequences l_primes(batch_size);
  size_t max_u_prime = 0;
  Status l_p_ret =
      PopulateLPrimesHMM(preprocess_collapse_repeated, batch_size, num_classes,
                      seq_len, labels, &max_u_prime, &l_primes);
  if (!l_p_ret.ok()) {
    return l_p_ret;
  }

  // Convert priors from tensor to vector
  Array priors_vec(num_classes);
  for (int i = 0; i < num_classes; i++) {
     priors_vec(i) = priors(i);
  }

  // Convert transition_probs from tensor to matrix
  Matrix t_p(num_classes, 2);
  for (int i=0; i < num_classes; i++) {
     t_p(i, 0) = transition_probs(i, 0);
     t_p(i, 1) = transition_probs(i, 1);
  }

  // Convert language transition_probs from tensor to matrix
  Matrix l_t_p(num_classes, num_classes + 1); // Also transition from blank to end
  for (int i=0; i < num_classes; i++) {
    for (int j=0; j < num_classes + 1; j++) {
        l_t_p(i, j) = lang_transition_probs(i, j);
    }
  }

  Eigen::MatrixXf d_priors_batch(batch_size, num_classes);
  Eigen::MatrixXf d_t_p_batch(batch_size * num_classes, 2);
  d_priors_batch.setConstant(0);
  d_t_p_batch.setConstant(0);

  // Process each item in a batch in parallel, using at most kMaxThreads.
  auto ComputeLossAndGradients = [this, num_classes, &l_primes, &seq_len,
                                  &inputs, &priors_vec, &t_p, &l_t_p, requires_backprop,
                                  hmm_merge_repeated, &loss, &gradients, &d_priors_batch, &d_t_p_batch, batch_size](
      int64 start_row, int64 limit_row) {
    for (int b = start_row; b < limit_row; b++) {

      if (seq_len(b) == 0) {
        continue;
      }

      // For each batch element, log(alpha) and log(beta).
      //   row size is: u_prime == l_prime.size()
      //   col size is: seq_len[b] - output_delay_
      const std::vector<int>& l_prime = l_primes[b];

      Matrix log_alpha_b(l_prime.size(), seq_len(b) - this->output_delay_);
      Matrix log_beta_b(l_prime.size(), seq_len(b) - this->output_delay_);
      Matrix log_tilde_alpha_b(num_classes, seq_len(b) - this->output_delay_);
      Matrix log_tilde_beta_b(num_classes, seq_len(b) - this->output_delay_);

      int num_maxes = (seq_len(b) - this->output_delay_) / T_maxes;
      // Plus 1 to avoid empty arrays. We don't access this index.
      Array log_alpha_b_maxes(num_maxes + 1);
      Array log_beta_b_maxes(num_maxes + 1);
      Array log_tilde_alpha_b_maxes(num_maxes + 1);
      Array log_tilde_beta_b_maxes(num_maxes + 1);

      // Work matrices, pre-allocated to the size required by this batch item.
      Matrix y(num_classes, seq_len(b));
      Eigen::MatrixXf dy;
      if (requires_backprop) dy = Eigen::MatrixXf::Zero(y.rows(), y.cols());

      Eigen::ArrayXf d_priors;
      if (requires_backprop) d_priors = Eigen::ArrayXf::Zero(num_classes);

      Eigen::MatrixXf d_t_p;
      if (requires_backprop) d_t_p = Eigen::MatrixXf::Zero(t_p.rows(), t_p.cols());

      // For this batch, we'll only work with this shortened sequence_length.
      Matrix y_b = y.leftCols(seq_len(b));

      // Convert label from DistBelief
      // y, prob are in num_classes x seq_len(b)
      // Output activations.
      Array y_b_col;
      for (int t = 0; t < seq_len(b); t++) {
        // Calculate the softmax of y_b.  Use double precision
        // arithmetic for the sum.
        double max_coeff = inputs[t].row(b).maxCoeff();
        y_b_col = (inputs[t].row(b).array().template cast<double>() - max_coeff).exp();
        y_b.col(t) = y_b_col / y_b_col.sum();
      }

      // Subtract priors from output activations
      for (int t=0; t < seq_len(b); t++) {
          y_b.col(t) = (y_b.col(t).array().log().matrix() - priors_vec.array().log().matrix());
      }

      CalculateAlpha(l_prime, y_b, t_p, l_t_p, priors_vec, hmm_merge_repeated, &log_alpha_b, &log_alpha_b_maxes);
      CalculateBeta(l_prime, y_b, t_p, l_t_p, hmm_merge_repeated, &log_beta_b, &log_beta_b_maxes);

      CalculateTildeAlpha(y_b, t_p, l_t_p, priors_vec, hmm_merge_repeated, &log_tilde_alpha_b, &log_tilde_alpha_b_maxes);
      CalculateTildeBeta(y_b, t_p, l_t_p, hmm_merge_repeated, &log_tilde_beta_b, &log_tilde_beta_b_maxes);

      double log_beta_maxes_sum = 0.0;
      double log_tilde_beta_maxes_sum = 0.0;
      for (int t = 0; t < num_maxes; ++t) {
        log_beta_maxes_sum += log_beta_b_maxes(t);
        log_tilde_beta_maxes_sum += log_tilde_beta_b_maxes(t);
      }

      Array log_p_z_x(seq_len(b));
      log_p_z_x.setConstant(kLogZero);

      for (int t = 0; t < seq_len(b); ++t) {
          for (int u = 0; u < l_prime.size(); ++u) {
            log_p_z_x(t) = LogSumExp(log_p_z_x(t), log_alpha_b(u, t) + log_beta_b(u, t));
          }
      }

      Array log_p_z_x_den(seq_len(b));
      log_p_z_x_den.setConstant(kLogZero);

      for (int t = 0; t < seq_len(b); ++t) {
          for (int u = 0; u < num_classes; ++u) {
            log_p_z_x_den(t) = LogSumExp(log_p_z_x_den(t), log_tilde_alpha_b(u, t) + log_tilde_beta_b(u, t));
          }
      }
        
	  (*loss)(b) = (float) -(log_p_z_x(0) + log_beta_maxes_sum) + log_p_z_x_den(0) + log_tilde_beta_maxes_sum;  // Use negative log loss for display.

      if (requires_backprop) {
        // Gradients with respect to input activations.
        // Calculate gradient.
        dy.setZero();
        Array alpha_beta_over_t_num(num_classes);

        CalculateGradientsNumerator(l_prime, y_b, log_alpha_b, log_alpha_b_maxes, log_beta_b, log_beta_b_maxes,
         log_p_z_x, priors_vec, t_p, l_t_p, &dy, &d_priors, &d_t_p, &alpha_beta_over_t_num, log_beta_maxes_sum);

        CalculateGradientsDenominator(y_b, log_tilde_alpha_b, log_tilde_alpha_b_maxes, log_tilde_beta_b, log_tilde_beta_b_maxes,
         log_p_z_x_den, priors_vec, t_p, l_t_p, alpha_beta_over_t_num, &dy, &d_priors, &d_t_p, log_tilde_beta_maxes_sum);

        // Convert gradient for current sample to DistBelief.
        for (int t = 0; t < seq_len(b); t++) {
          (*gradients)[t].row(b).array() = dy.col(t);
        }
        for (int i=0; i < num_classes; i++) {
            d_priors_batch(b, i) = d_priors(i) / batch_size; // Averaging over all batches
        }
        int batch_offset = b * num_classes;
        for (int i=0; i < num_classes; i++) {
          d_t_p_batch(batch_offset + i, 0) = d_t_p(i, 0) / batch_size; // Averaging over all batches
          d_t_p_batch(batch_offset + i, 1) = d_t_p(i, 1) / batch_size; // Averaging over all batches
        }

      }
    }  // for (int b = ...
  };
  if (workers) {
    // *Rough* estimate of the cost for one item in the batch.
    // Numerator's Forward, Backward: O((T - U) * U), Gradients: O(T * (U + L)).
    // Denominator's Forward, Backward: O(T * L * L), Gradients: O(T * L * L).
    // softmax: T * L * (Cost(Exp) + Cost(Div)) +
    // priors subtraction: T * L * (Cost(Add) + 2 * Cost(Log))
    // fwd,bwd numerator: 2 * (T - U) * U * (Cost(LogSumExp) + 3 * Cost(Log) + 4 * Cost(Add)) +
    // fwd,bwd denominator: 2 * T * L * (Cost(Log) + 2 * Cost(Add) + L * (Cost(LogSumExp) + 3 * Cost(Add) + 2 * Cost(Log))) +
    // grad numerator: T * (U * (10 * Cost(Add) + Cost(Log) + 3 * Cost(LogSumExp)) + L * (Cost(Add) + Cost(Exp))) + 2 * L * Cost(Exp) +
    // grad denominator: T * L * (Cost(Exp) + 8 * Cost(Add) + 2 * Cost(LogSumExp) + L * (5 * Cost(Add) + 4 * Cost(Exp) + 2 * Cost(Log))) + L * (4 * Cost(Exp) + 5 * Cost(Add) + 2 * Cost(Log)) +
    // normalization numerator: (T / T_maxes) * U * Cost(Add) +
    // normalization denominator: (T / T_maxes) * L * Cost(Add)
    const int64 cost_exp = Eigen::internal::functor_traits<
        Eigen::internal::scalar_exp_op<double>>::Cost;
    const int64 cost_log = Eigen::internal::functor_traits<
        Eigen::internal::scalar_log_op<double>>::Cost;
    const int64 cost_log_sum_exp =
        Eigen::TensorOpCost::AddCost<double>() + cost_exp + cost_log;
    const int64 cost_add = Eigen::TensorOpCost::AddCost<double>();
    const int64 cost =
        max_seq_len * num_classes *
            (cost_exp + Eigen::TensorOpCost::DivCost<double>()) +
        max_seq_len * num_classes *
            (cost_add + 2 * cost_log) +
        2 * (max_seq_len - max_u_prime) * (max_u_prime) * (cost_log_sum_exp + 3 * cost_log + 4 * cost_add) +
        2 * max_seq_len * num_classes * (cost_log + 2 * cost_add + num_classes * (cost_log_sum_exp + 3 * cost_add + 2 * cost_log)) +
        max_seq_len * (max_u_prime * (10 * cost_add + cost_log + 3 * cost_log_sum_exp) + num_classes * (cost_add + cost_exp)) + 2 * num_classes * cost_exp +
        max_seq_len * num_classes * (cost_exp + 8 * cost_add + 2 * cost_log_sum_exp + num_classes * (5 * cost_add + 4 * cost_exp + 2 * cost_log)) + num_classes * (4 * cost_exp + 5 * cost_add + 2 * cost_log) +
        (max_seq_len / T_maxes) * max_u_prime * cost_add +
        (max_seq_len / T_maxes) * num_classes * cost_add;

    Shard(workers->num_threads, workers->workers, batch_size, cost,
          ComputeLossAndGradients);
  } else {
    ComputeLossAndGradients(0, batch_size);
  }
  for (int b=0; b < batch_size; b++) {
    int batch_offset = b * num_classes;
    for (int l=0; l < num_classes; l++) {
        (*priors_gradient)(l) += d_priors_batch(b, l);
        (*trans_gradient)(l, 0) += d_t_p_batch(batch_offset + l, 0);
        (*trans_gradient)(l, 1) += d_t_p_batch(batch_offset + l, 1);
    }
  }
  return Status::OK();
}

template <typename Vector>
Status HmmMmiLossCalculator::PopulateLPrimesHMM(bool preprocess_collapse_repeated,
                                          int batch_size, int num_classes,
                                          const Vector& seq_len,
                                          const LabelSequences& labels,
                                          size_t* max_u_prime,
                                          LabelSequences* l_primes) const {
  // labels is a Label array of size batch_size
  if (labels.size() != batch_size) {
    return errors::InvalidArgument("labels.size() != batch_size: ",
                                   labels.size(), " vs. ", batch_size);
  }

  *max_u_prime = 0;  // keep track of longest l' modified label sequence.
  for (int b = 0; b < batch_size; b++) {
    // Assume label is in Label proto
    const std::vector<int>& label = labels[b];
    if (label.size() == 0) {
      return errors::InvalidArgument("Labels length is zero in batch ", b);
    }

    // If debugging: output the labels coming into training.
    //
    VLOG(2) << "label for batch: " << b << ": " << str_util::Join(label, " ");

    // Target indices, length = U.
    std::vector<int> l;

    // Convert label from DistBelief
    bool finished_sequence = false;
    for (int i = 0; i < label.size(); ++i) {
      if (i == 0 || !preprocess_collapse_repeated || label[i] != label[i - 1]) {
        if (label[i] >= num_classes) {
          finished_sequence = true;
        } else {
          if (finished_sequence) {
            // Saw an invalid sequence with non-null following null
            // labels.
            return errors::InvalidArgument(
                "Saw a non-null label (index >= num_classes) "
                "following a ",
                "null label, batch: ", b, " num_classes: ", num_classes,
                " labels: ", str_util::Join(l, ","));
          }
          l.push_back(label[i]);
        }
      }
    }

    // Make sure there is enough time to output the target indices.
    int time = seq_len(b) - output_delay_;
    int required_time = label.size();
    for (int l_i : l) {
      if (l_i < 0) {
        return errors::InvalidArgument(
            "All labels must be nonnegative integers, batch: ", b, " labels: ",
            str_util::Join(l, ","));
      } else if (l_i >= num_classes) {
        return errors::InvalidArgument(
            "No label may be greater than num_classes. ", "num_classes: ",
            num_classes, ", batch: ", b, " labels: ", str_util::Join(l, ","));
      }
    }
    if (required_time > time) {
      return errors::InvalidArgument(
          "Not enough time for target transition sequence ("
          "required: ",
          required_time, ", available: ", time,
          "), skipping data instance in batch: ", b);
    }

    // Convert l to l_prime
    GetLPrimeIndicesHMM(l, &l_primes->at(b));
    *max_u_prime = std::max(*max_u_prime, l_primes->at(b).size());
  }
  return Status::OK();
}

void HmmMmiLossCalculator::CalculateAlpha(
    const std::vector<int>& l_prime, const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, const Array& priors, bool hmm_merge_repeated,
    Matrix* log_alpha, Array* log_alpha_maxes) const {
  // Number of cols is the number of time steps = number of cols in target
  // after the output delay.
  log_alpha->setConstant(kLogZero);
  log_alpha_maxes->setZero();

  int U = l_prime.size();
  int T = log_alpha->cols();
  CHECK_EQ(U, log_alpha->rows());

  // Transition probability from start to blank is 1.0
  log_alpha->coeffRef(0, 0) = y(l_prime[0], output_delay_); 

  for (int t = 1; t < T; ++t) {
    // If there is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_alpha(u, t) continue to
    // be kLogZero.
    for (int u = std::max(0, U - (T - t)); u < std::min(U, t + 1);
         ++u) {

      // Begin recursion of alpha variable
      double sum_log_alpha = kLogZero;
      int l = l_prime[u];

      // Self transition
      sum_log_alpha = log_alpha->coeff(u, t - 1) + log(t_p(l, 0));

      // Transition out
      if (u > 0) {
          sum_log_alpha =
              LogSumExp(sum_log_alpha, log_alpha->coeff(u - 1, t - 1) + log(t_p(l_prime[u - 1], 1)) + log(l_t_p(l_prime[u - 1], l)));
      }

      // Multiply the summed alphas with the activation log probability (which is already subtracted by the prior).
      log_alpha->coeffRef(u, t) =
          y(l_prime[u], output_delay_ + t) + sum_log_alpha;
    }

    // Normalization
    if (t % T_maxes == 0) {
        int t_maxes = t / T_maxes - 1;
        log_alpha_maxes->coeffRef(t_maxes) = log_alpha->col(t).maxCoeff();
        log_alpha->col(t) = log_alpha->col(t).array() - log_alpha_maxes->coeff(t_maxes);
    }

  }
}

void HmmMmiLossCalculator::CalculateBeta(
    const std::vector<int>& l_prime, const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, bool hmm_merge_repeated,
    Matrix* log_beta, Array* log_beta_maxes) const {
  // Number of cols is the number of time steps =  number of cols in target.

  log_beta->setConstant(kLogZero);
  log_beta_maxes->setZero();
  int T = log_beta->cols();
  int U = l_prime.size();
  CHECK_EQ(U, log_beta->rows());
  int L = y.rows(); // num_classes
  int end_index = L;

  // Transition from final blank to end
  log_beta->coeffRef(U - 1, T - 1) = log(t_p(l_prime[U - 1], 1)) + log(l_t_p(blank_index_, end_index));


  for (int t = T - 1 - 1; t >= 0; --t) {
    // If there is not enough time to output the remaining labels or
    // some labels have been skipped, then let log_beta(u, t) continue to
    // be kLogZero.
    for (int u = std::max(0, U - (T - t)); u < std::min(U, t + 1); ++u) {

      // Begin recursion of beta variable
      double sum_log_beta = kLogZero;
      int l = l_prime[u];

      // Self transition
      sum_log_beta = log_beta->coeff(u, t + 1) + log(t_p(l, 0)) + y(l, output_delay_ + t + 1);

      // Transition out
      if (u + 1 < U) {
          sum_log_beta =
              LogSumExp(sum_log_beta, log_beta->coeff(u + 1, t + 1) + log(t_p(l, 1)) + log(l_t_p(l, l_prime[u + 1])) +
                                                                  y(l_prime[u + 1], output_delay_ + t + 1));
      }

      // Update beta variable at u, t.
      log_beta->coeffRef(u, t) = sum_log_beta;
    }

    // Normalization
    if ((t > 0) && (t % T_maxes == 0)) {
        int t_maxes = t / T_maxes - 1;
        log_beta_maxes->coeffRef(t_maxes) = log_beta->col(t).maxCoeff();
        log_beta->col(t) = log_beta->col(t).array() - log_beta_maxes->coeff(t_maxes);
    }
  }
}


void HmmMmiLossCalculator::CalculateTildeAlpha(
    const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, const Array& priors, bool hmm_merge_repeated,
    Matrix* log_alpha, Array* log_alpha_maxes) const {
  // Number of cols is the number of time steps = number of cols in target
  // after the output delay.
  log_alpha->setConstant(kLogZero);
  log_alpha_maxes->setZero();

  int T = log_alpha->cols();
  int L = y.rows(); // num_classes

  // Initial tilde alpha values

  // From start to blank
  log_alpha->coeffRef(blank_index_, 0) = y(blank_index_, output_delay_);

  for (int t = 1; t < T; ++t) {

    for (int c = 0; c < L; ++c) {

      // Begin recursion of alpha variable
      double sum_log_alpha = kLogZero;

      // Self transition
      sum_log_alpha = log_alpha->coeff(c, t - 1) + log(t_p(c, 0));

      // Out transition
      for (int hc = 0; hc < L; ++hc) {
        if (l_t_p(hc, c) != 0) {
            sum_log_alpha =
                    LogSumExp(sum_log_alpha, log_alpha->coeff(hc, t - 1) + log(t_p(hc, 1)) + log(l_t_p(hc, c)));
        }
      }

      // Multiply the summed alphas with the activation log probability (which is already subtracted by the prior).
      log_alpha->coeffRef(c, t) = y(c, output_delay_ + t) + sum_log_alpha;

    }

    // Normalization
    if (t % T_maxes == 0) {
        int t_maxes = t / T_maxes - 1;
        log_alpha_maxes->coeffRef(t_maxes) = log_alpha->col(t).maxCoeff();
        log_alpha->col(t) = log_alpha->col(t).array() - log_alpha_maxes->coeff(t_maxes);
    }
  }
}

// Calculates the beta(t, u) as described in (GravesTh) Section 7.3.
void HmmMmiLossCalculator::CalculateTildeBeta(
    const Matrix& y, const Matrix& t_p, const Matrix& l_t_p, bool hmm_merge_repeated,
    Matrix* log_beta, Array* log_beta_maxes) const {
  // Number of cols is the number of time steps =  number of cols in target.

  log_beta->setConstant(kLogZero);
  log_beta_maxes->setZero();
  int T = log_beta->cols();
  int L = y.rows(); // num_classes
  int end_index = L;

  // Initial tilde beta values

  // From blank to end
  log_beta->coeffRef(blank_index_, T - 1) = log(t_p(blank_index_, 1)) + log(l_t_p(blank_index_, end_index));

  for (int t = T - 1 - 1; t >= 0; --t) {

    for (int c = 0; c < L; ++c) {

      // Begin recursion of beta variable
      double sum_log_beta = kLogZero;

      // Self transition
      sum_log_beta = log_beta->coeff(c, t + 1) + log(t_p(c, 0)) + y(c, output_delay_ + t + 1);

      // Out transition
      for (int hc = 0; hc < L; ++hc) {
        if (l_t_p(c, hc) != 0) {
            sum_log_beta =
                LogSumExp(sum_log_beta, log_beta->coeff(hc, t + 1) + log(t_p(c, 1)) + log(l_t_p(c, hc)) + y(hc, output_delay_ + t + 1));
        }
      }

      // Update beta variable at c, t.
      log_beta->coeffRef(c, t) = sum_log_beta;

    }

    // Normalization
    if ((t > 0) && (t % T_maxes == 0)) {
        int t_maxes = t / T_maxes - 1;
        log_beta_maxes->coeffRef(t_maxes) = log_beta->col(t).maxCoeff();
        log_beta->col(t) = log_beta->col(t).array() - log_beta_maxes->coeff(t_maxes);
    }
  }
}


void HmmMmiLossCalculator::CalculateGradientsNumerator(const std::vector<int>& l_prime,
                                          const Matrix& y,
                                          Matrix& log_alpha,
                                          const Array& log_alpha_maxes,
                                          Matrix& log_beta,
                                          const Array& log_beta_maxes,
                                          const Array& log_p_z_x,
                                          const Array& priors,
                                          const Matrix& t_p,
                                          const Matrix& l_t_p,
                                          Eigen::MatrixXf* dy, Eigen::ArrayXf* d_priors, Eigen::MatrixXf* d_t_p,
                                          Array* alpha_beta_over_t_num,
                                          double log_beta_maxes_sum) const {
  // Only working with the leftmost part of dy for this batch element.
  auto dy_b = dy->leftCols(y.cols());

  // It is possible that no valid path is found if the activations for the
  // targets are zero.
  if (log_p_z_x(0) == kLogZero) {
    LOG(WARNING) << "No valid path found.";
    dy_b = y.template cast<float>();
    return;
  }

  int L = y.rows();
  int T = y.cols();
  int U = l_prime.size();
  int end_index = L;

  alpha_beta_over_t_num->setConstant(kLogZero);
  Matrix trans_alpha_beta(L, 2);
  trans_alpha_beta.setConstant(kLogZero);

  for (int t = 0; t < T - output_delay_; ++t) {
    Array alpha_beta(L);
    alpha_beta.setConstant(kLogZero);

    for (int u = 0; u < U; ++u) {
      int l = l_prime[u];
      alpha_beta[l] = LogSumExp(alpha_beta[l], log_alpha(u, t) + log_beta(u, t));
      alpha_beta_over_t_num->coeffRef(l) = LogSumExp(alpha_beta_over_t_num->coeff(l), log_alpha(u, t) + log_beta(u, t) - log_p_z_x(t)); // Normalize prior's graidents with log_p_z_x here

      // Normalize transition's gradients. Normalization might be different at each t (since we use alpha and beta at different t's).
      double trans_normalization_factor = log_p_z_x(t);
      if ((t > 0) && (t % T_maxes == 0)) {
        int t_maxes = t / T_maxes - 1;
        trans_normalization_factor += log_beta_maxes(t_maxes);
      }

      if (t < T - 1) {
        trans_alpha_beta(l, 0) = LogSumExp(trans_alpha_beta(l, 0), log_alpha(u, t) + log_beta(u, t + 1) - trans_normalization_factor + y(l, output_delay_ + t + 1));
        if (u < U - 1) {
            trans_alpha_beta(l, 1) = LogSumExp(trans_alpha_beta(l, 1),
                                               log_alpha(u, t) + log(l_t_p(l, l_prime[u + 1])) + log_beta(u + 1, t + 1) - trans_normalization_factor + y(l_prime[u + 1], output_delay_ + t + 1));
        }
      }
    }

    for (int l = 0; l < L; ++l) {
      double negative_term = exp(alpha_beta[l] - log_p_z_x(t));

      dy_b(l, output_delay_ + t) = (float) - negative_term;
    }

  }
  trans_alpha_beta(l_prime[U - 1], 1) = LogSumExp(trans_alpha_beta(l_prime[U - 1], 1), log_alpha(U - 1, T - 1)  - log_p_z_x(T - 1) + log(l_t_p(l_prime[U - 1], end_index)));

  // Transition probabilities gradients
  for (int l = 0; l < L; ++l) {
    d_t_p->coeffRef(l, 0) = (float) - exp(trans_alpha_beta(l, 0));

    d_t_p->coeffRef(l, 1) = (float) - exp(trans_alpha_beta(l, 1));
  }

}

void HmmMmiLossCalculator::CalculateGradientsDenominator(
                                          const Matrix& y,
                                          Matrix& log_alpha,
                                          const Array& log_alpha_maxes,
                                          Matrix& log_beta,
                                          const Array& log_beta_maxes,
                                          const Array& log_p_z_x,
                                          const Array& priors,
                                          const Matrix& t_p,
                                          const Matrix& l_t_p,
                                          const Array& alpha_beta_over_t_num,
                                          Eigen::MatrixXf* dy, Eigen::ArrayXf* d_priors, Eigen::MatrixXf* d_t_p,
                                          double log_beta_maxes_sum) const {
  // Only working with the leftmost part of dy for this batch element.
  auto dy_b = dy->leftCols(y.cols());

  // It is possible that no valid path is found if the activations for the
  // targets are zero.
  if (log_p_z_x(0) == kLogZero) {
    LOG(WARNING) << "No valid path found.";
    dy_b = y.template cast<float>();
    return;
  }

  int L = y.rows();
  int T = y.cols();
  int end_index = L;

  Array alpha_beta_over_t(L);
  alpha_beta_over_t.setConstant(kLogZero);
  Matrix trans_alpha_beta(L, 2);
  trans_alpha_beta.setConstant(kLogZero);

  for (int t = 0; t < T - output_delay_; ++t) {

    // Normalize transition's gradients. Normalization might be different at each t (since we use alpha and beta at different t's).
    double trans_normalization_factor = log_p_z_x(t);
    if ((t > 0) && (t % T_maxes == 0)) {
      int t_maxes = t / T_maxes - 1;
      trans_normalization_factor += log_beta_maxes(t_maxes);
    }

    for (int c = 0; c < L; ++c) {

      // Derivative of network's outputs
      double negative_term = exp(log_alpha(c, t) + log_beta(c, t) - log_p_z_x(t));

      float new_deriv = (float) - negative_term;
      dy_b(c, output_delay_ + t) = (dy_b(c, output_delay_ + t) - new_deriv); 
      alpha_beta_over_t[c] = LogSumExp(alpha_beta_over_t[c], log_alpha(c, t) + log_beta(c, t) - log_p_z_x(t)); // Normalize prior's graidents with log_p_z_x here

      if (t < T - 1) {

        // Self transition
        trans_alpha_beta(c, 0) = LogSumExp(trans_alpha_beta(c, 0), log_alpha(c, t) + log_beta(c, t + 1) - trans_normalization_factor + y(c, output_delay_ + t + 1));

        // Out transition
        for (int hc = 0; hc < L; ++hc) {

            if (l_t_p(c, hc) != 0) {
                trans_alpha_beta(c, 1) = LogSumExp(trans_alpha_beta(c, 1),
                                                        log_alpha(c, t) + log_beta(hc, t + 1) - trans_normalization_factor +
                                                        log(l_t_p(c, hc)) + y(hc, output_delay_ + t + 1));
            }
        }
      }
    }
  }

  // Transitions from blank to end
  trans_alpha_beta(blank_index_, 1) = LogSumExp(trans_alpha_beta(blank_index_, 1), log_alpha(blank_index_, T - 1) + log(l_t_p(blank_index_, end_index)) - log_p_z_x(T - 1));

  float new_deriv;

  // Priors gradients
  for (int l = 0; l < L; ++l) {
    d_priors->coeffRef(l) = (exp(alpha_beta_over_t_num[l] - log(priors(l))) - exp(alpha_beta_over_t[l] - log(priors(l))));
  }

  // Transition probabilities gradients
  for (int l = 0; l < L; ++l) {
    new_deriv = (float) -exp(trans_alpha_beta(l, 0));
    d_t_p->coeffRef(l, 0) = (d_t_p->coeff(l, 0) - new_deriv);
    new_deriv = (float) -exp(trans_alpha_beta(l, 1));
    d_t_p->coeffRef(l, 1) = (d_t_p->coeff(l, 1) - new_deriv);

  }
}

void HmmMmiLossCalculator::GetLPrimeIndicesHMM(const std::vector<int>& l,
                                         std::vector<int>* l_prime) const {
  // Assumption is that l_prime is empty.
  l_prime->reserve(l.size());

  for (auto label : l) {
    l_prime->push_back(label);
  }
}

}  // namespace hmm
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_HMM_MMI_LOSS_CALCULATOR_H_
