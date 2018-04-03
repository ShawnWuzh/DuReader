
# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the Pointer Network for selecting answer spans, as described in:
https://openreview.net/pdf?id=B1-q5Pqxl
"""

import tensorflow as tf
import tensorflow.contrib as tc


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2])) # 0： batch_size 1: sequence_length 2: hidden_unit
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)        # initial time step is time0
    if initial_state is not None:
        s0 = initial_state
    else:
        s0 = cell.zero_state(batch_size, dtype=tf.float32) # so with the shape(batch_size,state_size)
    f0 = tf.zeros([batch_size], dtype=tf.bool)
    # f0 is of the type: [0,0,0,......,0]
    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)   # read data from time to time with shape(batch_size, 1)
        scores, cur_state = cell(cur_x, prev_s)

        # copy through
        scores = tf.where(finished, tf.zeros_like(scores), scores)
        # zeros like is about replace all zeros with the same type as the scores
        if isinstance(cell, tc.rnn.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]

    _, state, emit_ta, _ = tf.while_loop(
        cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
        # logical not: just not
        # reduce_all: computes the "logical and" of elements across dimensions of a tensor
        body=loop_fn,
        loop_vars=(t0, s0, emit_ta, f0),
        parallel_iterations=32,
        swap_memory=False)
  #emit_ta of shape(time_index,passage_lengh,1)
    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state


def attend_pooling(pooling_vectors, ref_vector, hidden_size, scope=None):
    """
    Applies attend pooling to a set of vectors according to a reference vector.
    Args:
        pooling_vectors: the vectors to pool (question_vectors)   (batch_size,question_length, 2 * hidden_size)
        ref_vector: the reference vector    (random_attn_vector)    [1,hidden_size]
        hidden_size: the hidden size for attention function
        scope: score name
    Returns:
        the pooled vector
    reviewed
    """
    with tf.variable_scope(scope or 'attend_pooling'):
        U = tf.tanh(tc.layers.fully_connected(pooling_vectors, num_outputs=hidden_size,
                                              activation_fn=None, biases_initializer=None)  #(batch_size,question_length,hidden_size)
                    + tc.layers.fully_connected(tf.expand_dims(ref_vector, 1),
                                                num_outputs=hidden_size,          #(1,1,hidden_size)
                                                activation_fn=None))

        # (batch_size,question_length,hidden_size)
        logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
        # (batch_size,question_length,1)
        scores = tf.nn.softmax(logits, 1)
        # (batch_size,question_length,1)
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
        # (batch_size,hidden_size)
    return pooled_vector


class PointerNetLSTMCell(tc.rnn.LSTMCell):
    """
    Implements the Pointer Network Cell
    reviewed
    """
    def __init__(self, num_units, context_to_point):
        # context_to_point is the passage vectors
        super(PointerNetLSTMCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_point = context_to_point
        self.fc_context = tc.layers.fully_connected(self.context_to_point,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)
        # self.fc_context is of the shape of (batch_size,passage_all_length,num_units)
        # Attention Attention: the output of  fully connected layer has the same shape of the input
    def __call__(self, inputs, state, scope=None):
        # inputs with shape (batch_size,1)
        (c_prev, m_prev) = state
        # c_prev with (batch_size,state_size)
        with tf.variable_scope(scope or type(self).__name__):
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(m_prev,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None),
                                         1))
            # U with the shape(batch_size,all_passage_length,num_units)
            logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
            # logits with the shape of (batch_size,all_passage_length,1)
            scores = tf.nn.softmax(logits, 1)
            # scores is of the shape of (batch_size, all_passage_length,1)
            attended_context = tf.reduce_sum(self.context_to_point * scores, axis=1)
            # attended_context is of the shape of the (batch_size,2 * hidden_units)
            lstm_out, lstm_state = super(PointerNetLSTMCell, self).__call__(attended_context, state)
        return tf.squeeze(scores, -1), lstm_state
    # squeeze: removes dimensions of size 1 from the shape a tensor


class PointerNetDecoder(object):
    """
    Implements the Pointer Network
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def decode(self, passage_vectors, passage_ranking_vectors, question_vectors, batch_size,init_with_question=True):
        """
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Args:
            passage_vectors: the encoded passage vectors
            question_vectors: the encoded question vectors
            init_with_question: if set to be true,
                             we will use the question_vectors to init the state of Pointer Network
        Returns:
            the probs of evary position to be start and end of the answer
        """
        with tf.variable_scope('pn_decoder'):
            fake_inputs = tf.zeros([tf.shape(passage_vectors)[0], 2, 1])  # not used
            # 2 stands for 2 steps, one step for start_prob , another for end_prob
            sequence_len = tf.tile([2], [tf.shape(passage_vectors)[0]])   #
            # each position of the passage needs a start and end output
            # sequence length is two

            random_attn_vector = tf.Variable(tf.random_normal([1, self.hidden_size]),
                                             trainable=True, name="random_attn_vector")
            pooled_question_rep = tc.layers.fully_connected(
                attend_pooling(question_vectors, random_attn_vector, self.hidden_size),
                num_outputs=self.hidden_size, activation_fn=None
            )
            if init_with_question:

                init_state = tc.rnn.LSTMStateTuple(pooled_question_rep, pooled_question_rep)
                # lstmstatetuple is a tuple storing two elements (c,h) c is the hidden state, and h is output
            else:
                init_state = None
            with tf.variable_scope('fw'):
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                fw_outputs, _ = custom_dynamic_rnn(fw_cell, fake_inputs, sequence_len, init_state)
            with tf.variable_scope('bw'):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                bw_outputs, _ = custom_dynamic_rnn(bw_cell, fake_inputs, sequence_len, init_state)
            start_prob = (fw_outputs[0:, 0, 0:] + bw_outputs[0:, 1, 0:]) / 2
            end_prob = (fw_outputs[0:, 1, 0:] + bw_outputs[0:, 0, 0:]) / 2
            matching_scores = passage_attention_to_question(pooled_question_rep,passage_ranking_vectors,batch_size,self.hidden_size)
            return start_prob, end_prob,matching_scores


def passage_attention_to_question(question_vectors,passage_encodes,batch_size,num_units):
    # passage_encodes is of the shape (batch_size,num_of_passage,passage_length,num_units)
    # question vectors is of the shape (batch_size, num_units)
    with tf.variable_scope('passage_ranking'):
        U = tf.tanh(tc.layers.fully_connected(passage_encodes, num_outputs=num_units,
                                              activation_fn=None, biases_initializer=None)
                    #(batch_size,num_of_passages,passage_length,hidden_size)
                    + tc.layers.fully_connected(tf.expand_dims(tf.expand_dims(question_vectors, 1),1),
                                                num_outputs=num_units,          #(batch_size,1,1,hidden_size)
                                                activation_fn=None))
        # num_of_passages = tf.shape(passage_encodes)[1]    # set the number of passages
        # (batch_size,num_of_passage,passage_length,hidden_size)
        logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
        # (batch_size,num_of_passage,passage_length,1)
        scores = tf.nn.softmax(logits, 2)
        # (batch_size,num_of_passage,passage_length,1)
        question_attention_passage = tf.reduce_sum(passage_encodes * scores, axis=2)
        # (batch_size,num_of_passage,num_units)
        concat_input = tf.concat([question_attention_passage,tf.tile(tf.expand_dims(tf.squeeze(question_vectors),1),
                                                                     [1,tf.shape(question_attention_passage)[1],1])],-1)
        # concat_input.set_shape([batch_size,num_of_passages, 300 + num_units])
        g = tf.tanh(tc.layers.fully_connected(tf.reshape(concat_input,[batch_size,-1,300 + num_units])
                                            ,num_outputs=num_units,activation_fn=None))
        logits = tc.layers.fully_connected(g,num_outputs=1,activation_fn=None)
        match_scores = tf.nn.softmax(logits,1)
        # (batch_size,num_of_passage,1)
        tf.squeeze(match_scores)
    return match_scores