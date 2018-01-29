#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

def depseparable_conv3v3(input_data,depthwise_filter, pointwise_filter,name):
    # input_data = tf.Variable(np.random.rand(1, 9, 9, 3), dtype=np.float32)
    # depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 4), dtype=np.float32)
    # pointwise_filter = tf.Variable(np.random.rand(1, 1, 12, 20), dtype=np.float32)
    y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME',name = name)
log_dir = 'F:'

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    return mnist


def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name =name)
def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x,W,name):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID',name=name)
def conv2d_same(x,W,name):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name)
def depseparable_conv(input_data,depthwise_filter, pointwise_filter,name):
    # input_data = tf.Variable(np.random.rand(1, 9, 9, 3), dtype=np.float32)
    # depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 1), dtype=np.float32)
    # pointwise_filter = tf.Variable(np.random.rand(1, 1, 3, 20), dtype=np.float32)
    y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME',name = name+"_separa")
    return y
def max_pool_2x2(x,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name = name)
def max_pool_2x2_same(x,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name = name)
def variable_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

def evaluate_pictures(n_epochs=20,batch_size=50):
    def loss(logits, labels):
        labels = tf.cast(labels, tf.int64) # 类型转换
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='cross_entropy') # 内部执行了softmax。
        tf.add_to_collection('losses', cross_entropy)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    mnist = load_data()
    train_set_x  = mnist.train.images
    train_set_y = mnist.train.labels
    test_set_x = mnist.test.images
    test_set_y = mnist.test.labels



    # 计算各数据集的batch个数
    n_train_batches = train_set_x.shape[0]
    n_test_batches  = test_set_x.shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_test_batches  = int(n_test_batches / batch_size)
    print("... building the model")

    # 搭建神经网络
    x = tf.placeholder(tf.float32, shape=[None, 784], name = 'input_x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name = 'label_y')
    keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
    x_images = tf.reshape(x, [-1, 28, 28, 1], name = 'x_tensor')
    tf.summary.image('input', x_images, 10)

    with tf.name_scope("conv1"):
        wd_cov1 = weight_variable([1, 1, 1, 1], 'wd')
        wp_cov1 = weight_variable([1, 1, 1, 6], 'wp')
        b_cov1 = bias_variable([6], name = 'b')
        h_cov1 = tf.nn.relu(depseparable_conv(x_images, wd_cov1,wp_cov1, name= 'conv1') + b_cov1,name = 'relu')
    with tf.name_scope("con1_incep3a1"):
        wd1 = weight_variable([1, 1, 6, 1], 'wd')
        wp1 = weight_variable([1, 1, 6, 4], 'wp')
        b1 = bias_variable([4], name = 'b')
        h1= tf.nn.relu(depseparable_conv(h_cov1, wd1,wp1, name= 'conv1') + b1,name = 'relu')
    with tf.name_scope("con2_incep3a1"):
        wd2 = weight_variable([3, 3, 4, 1], 'wd')
        wp2 = weight_variable([1, 1, 4, 4], 'wp')
        b2 = bias_variable([4], name='b')
        h2 = tf.nn.relu(depseparable_conv(h1, wd2,wp2, name= 'conv2') + b2, name='relu')

    with tf.name_scope("con1_incep3a2"):
        wd1_1 = weight_variable([1, 1, 6, 1], 'wd')
        wp1_1 = weight_variable([1, 1, 6, 4], 'wp')
        b1_1 = bias_variable([4], name = 'b')
        h1_1= tf.nn.relu(depseparable_conv(h_cov1, wd1_1,wp1_1, name= 'conv1') + b1_1,name = 'relu')
    with tf.name_scope("con2_incep3a2"):
        wd2_1 = weight_variable([3, 3, 4, 1], 'wd')
        wp2_1 = weight_variable([1, 1, 4, 8], 'wp')
        b2_1 = bias_variable([8], name='b')
        h2_1 = tf.nn.relu(depseparable_conv(h1_1, wd2_1,wp2_1, name='conv1_1') + b2_1, name='relu')

    with tf.name_scope("con1_incep3a3"):
        wd1_2 = weight_variable([1, 1, 6, 1], 'wd')
        wp1_2 = weight_variable([1, 1, 6, 12], 'wp')
        b1_2 = bias_variable([12], name='b')
        h2_2 = tf.nn.relu(depseparable_conv(h_cov1, wd1_2,wp1_2, name='conv1') + b1_2, name='relu')

    with tf.name_scope("max1_incep3a4"):
        h1_3= max_pool_2x2_same(h_cov1,name = 'maxpooling_conv2')
    with tf.name_scope("con2_incep3a4"):
        wd2_3 = weight_variable([1, 1, 6, 1], 'wd')
        wp2_3 = weight_variable([1, 1, 6, 8], 'wp')
        b2_3 = bias_variable([8], name='b')
        h2_3 = tf.nn.relu(depseparable_conv(h1_3, wd2_3,wp2_3, name='conv1_1') + b2_3, name='relu')
    incep = tf.concat([h2,h2_1,h2_2,h2_3],3)
    with tf.name_scope("conv2"):
        wd_cov2 = weight_variable([3, 3, 32, 1],name = 'wd_conv2')
        wp_cov2 = weight_variable([1, 1, 32, 16],name = 'wp_conv2')
        b_cov2 = bias_variable([16],name = 'b_conv2')
        h_cov2 = tf.nn.relu(depseparable_conv(incep, wd_cov2,wp_cov2, name = 'conv2') + b_cov2,name = 'relu_conv2')
        h_pool2 = max_pool_2x2(h_cov2,name = 'maxpooling_conv2')
    with tf.name_scope("conv3"):
        wd_cov3 = weight_variable([3, 3, 16, 1],name = 'wd_conv3')
        wp_cov3 = weight_variable([1, 1, 16, 8],name = 'wp_conv3')
        b_cov3 = bias_variable([8],name = 'b_conv3')
        h_cov3 = tf.nn.relu(depseparable_conv(h_pool2, wd_cov3,wp_cov3, name = 'conv3') + b_cov3,name = 'relu_conv3')
        h_pool3 = max_pool_2x2(h_cov3,name = 'maxpooling_conv3')
    with tf.name_scope("fc1"):
        h_pool2_reshape = tf.reshape(h_pool3, [-1, 7*7*8], name='cnn_fc_convert')
        w_fc1 = weight_variable([7*7*8,60],name = 'w_fc1')
        b_fc1 = bias_variable([60],name = 'b_fc1')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, w_fc1) + b_fc1,name= 'relu_fc1')
    with tf.name_scope("dropout"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name= 'dropout')
    with tf.name_scope("fc2"):
        w_fc2 = weight_variable([60, 10],name = 'w_fc2')
        b_fc2 = bias_variable([10],name = 'b_dc2')
        y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, w_fc2), b_fc2,name = 'y')
    with tf.name_scope("loss"):
        loss=loss(labels=y, logits=y_conv)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
    # 启动session
    sess=tf.Session()
    sess.run(tf.global_variables_initializer()) #初始化graph的参数

    best_validation_acc = 0
    epoch = 0

    print("... training")

    print(tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # summaries合并
    merged = tf.summary.merge_all()
    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter('F:/sum/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    while (epoch < n_epochs):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            summary,acc,_=sess.run([merged, accuracy,train_step],feed_dict={x: train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], keep_prob: 0.5})
            print('epoch %i, step %d,minibatch %i / %i, train acc %f' % (epoch, iter, minibatch_index + 1, n_train_batches,acc))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            train_writer.add_run_metadata(run_metadata, 'step%03d' % iter)
            train_writer.add_summary(summary, iter)
            # train_writer.add_summary(summary, iter)

            if (iter + 1) % 100 == 0:
                valid_acc=0
                for i in range(n_test_batches):
                    acc=sess.run([accuracy],feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
                    valid_acc =valid_acc+ acc[0]
                valid_acc=valid_acc/n_test_batches
                print('                         validation acc %g' %(valid_acc ))
                if valid_acc>best_validation_acc:
                    best_validation_acc=valid_acc
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy/accuracy"])
                    with tf.gfile.FastGFile('t001.pb', mode = 'wb') as f:
                        f.write(output_graph_def.SerializeToString())

    train_writer.close()
    print('Optimization complete.')
    test_acc=0;
    start_time=time.time()
    valid_acc=0
    print(start_time)
    for i in range(n_test_batches):
        valid_acc =valid_acc+ sess.run(accuracy,feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
    end_time=time.time()
    test_acc=valid_acc/n_test_batches
    print("test accuracy %g" % test_acc)
    print((end_time - start_time)*1000/60)

if __name__ == '__main__':
    evaluate_pictures()


