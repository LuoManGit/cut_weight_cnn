#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import pickle
import time
from weight import c2c_std,acc_w_c2f,acc_w_f2f

infile = 't101.pb'
outfile = 't102.pb'
def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(infile), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    return mnist


batch_size=100
n_epochs=5
mnist = load_data()
train_set_x  = mnist.train.images
train_set_y = mnist.train.labels
test_set_x = mnist.test.images
test_set_y = mnist.test.labels
n_train_batches = train_set_x.shape[0]
n_test_batches  = test_set_x.shape[0]
n_train_batches = int(n_train_batches / batch_size)
n_test_batches  = int(n_test_batches  / batch_size)
best_validation_acc = 0
epoch = 0
done_looping = False
with tf.Session() as sess:
    # create_graph()
    create_graph()
    print("... training")
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir='F:\\from github\\cut_weight_cnn\\graph', graph_def=sess.graph_def)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            _, summary, acc = sess.run([train_step, merged, accuracy], feed_dict={
                'input_x:0': train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                'label_y:0': train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], 'keep_prob:0': 0.5})
            print('epoch %i, step %d,minibatch %i/%i, train acc %g ' %
                  (epoch, iter, minibatch_index + 1, n_train_batches, acc))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            train_writer.add_run_metadata(run_metadata, 'step%03d' % iter)
            #train_writer.add_summary(summary, iter)

        valid_acc = 0
        for i in range(n_test_batches):
            acc = sess.run("accuracy/accuracy", feed_dict={'input_x:0': test_set_x[i * batch_size:(i + 1) * batch_size],
                                                'label_y:0': test_set_y[i * batch_size:(i + 1) * batch_size], 'keep_prob:0': 1})
            valid_acc = valid_acc + acc
        valid_acc = valid_acc / n_test_batches
        print('                         validation acc %g' % (valid_acc))
        if valid_acc > best_validation_acc:
            best_validation_acc = valid_acc
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names=["accuracy/accuracy"])
            with tf.gfile.FastGFile(outfile, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())

start_time = time.time()
valid_acc = 0
print(start_time)
for i in range(n_test_batches):
    valid_acc = valid_acc + sess.run("accuracy/accuracy", feed_dict={'input_x:0': test_set_x[i * batch_size:(i + 1) * batch_size],
                                                                     'label_y:0': test_set_y[i * batch_size:(i + 1) * batch_size],
                                                                     'keep_prob:0': 1})
end_time = time.time()
test_acc = valid_acc / n_test_batches
print("test accuracy %g" % test_acc)
print((end_time - start_time) * 1000 / 60)