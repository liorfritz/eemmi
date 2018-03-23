import tensorflow as tf
import hmm_ops

def mmi_loss(logits, sparse_labels, seq_lengths, num_labels, lang_transition_probs):


    # Priors
    unconstrained_priors = tf.Variable(tf.ones(num_labels) * (1 / float(num_labels)))
    priors = tf.nn.softmax(unconstrained_priors)

    # Transition Probabilities
    unconstrained_transition_probs = tf.Variable(tf.ones([num_labels, 2]) * (1 / float(2)))
    transition_probs = tf.nn.softmax(unconstrained_transition_probs)

    loss = hmm_ops.hmm_mmi_loss(logits, sparse_labels, tf.to_int32(seq_lengths),
                                    priors, transition_probs, lang_transition_probs)

    return tf.reduce_mean(loss), priors, transition_probs


def ctc_loss(logits, sparse_labels, seq_lengths):
    loss = tf.nn.ctc_loss(logits, sparse_labels, tf.to_int32(seq_lengths))

    return tf.reduce_mean(loss)



