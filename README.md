# End-to-End MMI
This is an implementation of the loss function from the paper [Simplified End-to-End MMI Training and Voting for ASR](https://arxiv.org/abs/1703.10356).

### What is included
Currently this repository includes callable functions of two loss functions: CTC and MMI. It also includes the source C++ files for the TensorFlow operator which calculates the MMI loss function (don't forget to build it first, as explained below).

All you need to do is integrate your code (which handles data and neural network architecture) with the supplied loss functions. Please see [this](https://github.com/vrenkens/tfkaldi) wonderful repository for advice on Kaldi+TensorFlow integration.

### Build the operator
This refers to TensorFlow 1.4. Please refer to [here](https://www.tensorflow.org/extend/adding_an_op) for updated versions.

    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
    g++ -std=c++11 -shared core/ops/hmm_mmi_ops.cc -o gen_hmm_mmi_ops.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2

### Using the `mmi_loss` function

The `utils/loss_functions.py` includes the CTC and MMI loss functions. You can use the CTC function as reference for how the inputs to the MMI loss function should be built.

There are two differences between the inputs of the loss functions:
1. `num_labels` - This holds the number of labels/classes. Should be identical to the size of the last dimension of `logits`.
2. `lang_transition_probs` - This is a language transition probabilities matrix, of a bi-grams model (refer to $q(c, \hat{c})$ in the paper). It should hold a numpy float matrix, of size [num_labels, num_labels + 1]. The rows refer to the current state ($c$), and the columns refer to the next state ($\hat{c}$). The last column denotes the ending state (*). You should build this matrix from simple counting on the training labels sequences.

(*) We need this extra column since the blank state (always final state) can be before all states. Therefore, there is some probability of a transition from blank to end.


### Todos

 - Supply code of NN architecture, and data handling (possibly EESEN+Kaldi integration)
 - Constantly check that the code supports new versions of TensorFlow

