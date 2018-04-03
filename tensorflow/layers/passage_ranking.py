# written by Zongheng WU for
# passage_ranking


import tensorflow as tf
import tensorflow.contrib as tc




def passage_attention_to_question(question_encodes,passage_encodes,num_units):
    # passage_encodes is of the shape of (batch_size,num_of_passages,passage_length,2*hidden_size)
    # question_encodes is of the shape of (batch_size,num_of_passages,question_length, 2*hidden_size)
    sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
    # passage_encodes (batch_size,num_of_all_passages,passage_length,embedding_dim)
    # question_encodes (batch_size,num_of_all_passages,question_length,embdding_dim)
    # sim_matrix (batch_size,num_of_all_passages,passage_length,question_length)
    b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, -1), 2), -1)
    # reduce_max (batch_size, passage_length)
    # b (batch_size, num_of_passages,1, passage_length)
    # b [32,1,5,11]
    # passage encodes [32,5,500,300]
    # passage_encodes (batch_size,num_of_passages,passage_length,embedding_dim)
    question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                    [1, 1,tf.shape(passage_encodes)[2], 1])

    concat_outputs = tf.concat([question_encodes,
                                tf.expand_dims(tf.reduce_sum(passage_encodes * question2context_attn,2),2)], 2)

    s = tf.tanh(tc.layers.fully_connected(concat_outputs,num_outputs=num_units,activation_fn=None),
                )
    # sj = V^t tanh
    logits = tc.layers.fully_connected(tf.reduce_sum(s,2),num_outputs=1,activation_fn= None)
    scores = tf.nn.softmax(logits, 1)
    # attention_passage = tf.reduce_sum(passage_encodes * scores,axis=1)
    # g = tf.layers.fully_connected(tf.tanh(tf.layers.fully_connected(tf.concat([attention_passage,question_encodes],-1))))
    # match_scores = tf.nn.softmax(g,1)
    return scores


