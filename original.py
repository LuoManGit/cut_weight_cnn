import tensorflow as tf
import numpy as np
import os
import pickle
import time
#from PIL import Image


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/root/ipythoncmd/cifar-100-python/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', 'root/ipythoncmd/mnist_logs', 'Summaries directory')

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def load_data(dataset_path):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    return mnist


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

def variable_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

def evaluate_pictures(n_epochs=20,batch_size=50,dataset='/root/ipythoncmd/quanti/cifar-10-batches-py'):
    def loss(logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    mnist = load_data(dataset)
    train_set_x  = mnist.train.images
    train_set_y = mnist.train.labels
    test_set_x = mnist.test.images
    test_set_y = mnist.test.labels


    #train_set_x, train_set_y = datasets[0]
    #test_set_x, test_set_y = datasets[1]

    # 计算各数据集的batch个数
    n_train_batches = train_set_x.shape[0]
    n_test_batches  = test_set_x.shape[0]
    n_train_batches = int(n_train_batches / batch_size)
    n_test_batches  = int(n_test_batches  / batch_size)
    print("... building the model")

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_images = tf.reshape(x, [-1, 28, 28, 1])
    print(x_images.name)
    W_cov1 = weight_variable([5, 5, 1, 20])
    print(W_cov1.name)
    b_cov1 = bias_variable([20])
    print(b_cov1.name)
    h_cov1 = tf.nn.relu(conv2d(x_images, W_cov1) + b_cov1)
    h_pool1 = max_pool_2x2(h_cov1)

    W_cov2 = weight_variable([5, 5, 20, 50])
    print(W_cov2.name)
    b_cov2 = bias_variable([50])
    print(b_cov2.name)
    h_cov2 = tf.nn.relu(conv2d(h_pool1, W_cov2) + b_cov2)
    h_pool2 = max_pool_2x2(h_cov2)

    W_fc1 = weight_variable([4 * 4 * 50, 500])
    print(W_fc1.name)
    b_fc1 = bias_variable([500])
    print(b_fc1.name)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 50])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([500, 10])
    print(W_fc2.name)
    b_fc2 = bias_variable([10])
    print(b_fc2.name)

    #y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
    # x_image = tf.transpose(tf.reshape(x, [-1, 1, 28, 28]),perm=[0,2,3,1])
    # w_conv1 = variable_weight_loss(shape=[1, 1, 1, 64],stddev=5e-2,w1=0.001)
    # b_conv1 = bias_variable([64])
    # h_pool1 = max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1)))
    #
    # w_conv2 = variable_weight_loss(shape=[5, 5, 64,128],stddev=5e-2,w1=0.001)
    # b_conv2 = bias_variable([128])
    # h_pool2 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, w_conv2) , b_conv2)))
    #
    # w_conv3 = variable_weight_loss(shape=[5, 5, 128, 256],stddev=5e-2,w1=0.001)
    # b_conv3 = bias_variable([256])
    # h_pool3 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2, w_conv3) , b_conv3)))
    #
    # w_conv4 = variable_weight_loss(shape=[5, 5, 256, 300],stddev=5e-2,w1=0.001)
    # b_conv4 = bias_variable([300])
    # h_pool4 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool3, w_conv4) , b_conv4)))
    #
    # w_conv5 = variable_weight_loss(shape=[3, 3, 300, 400],stddev=5e-2,w1=0.001)
    # b_conv5 = bias_variable([400])
    # h_pool5 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool4, w_conv5) , b_conv5)))
    #
    # h_fc1_drop = tf.nn.dropout(h_pool5, keep_prob)
    # w_fc1 = variable_weight_loss(shape=[1 * 1 * 400, 50],stddev=0.04,w1=0.004)
    # b_fc1 = bias_variable([50])
    # y_fc1 = tf.nn.bias_add(tf.matmul(tf.reshape(h_fc1_drop, [-1, 1 * 1 * 400]), w_fc1), b_fc1)
    #
    # h_fc2_drop = tf.nn.dropout(y_fc1, keep_prob)
    # W_fc2 =  variable_weight_loss(shape=[50,10],stddev=0.04,w1=0.004)
    # b_fc2 = bias_variable([10])
    # y_conv = tf.nn.bias_add(tf.matmul(h_fc2_drop, W_fc2) , b_fc2)

    loss=loss(labels=y, logits=y_conv)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    best_validation_acc = 0
    epoch = 0
    done_looping = False

    print("... training")
    print(x.name)
    print(y.name)
    print(keep_prob.name)
    print(accuracy.name)
    # print(w_conv1.name)
    # print(b_conv1.name)
    # print(w_conv2.name)
    # print(b_conv2.name)
    # print(w_conv3.name)
    # print(b_conv3.name)
    # print(w_conv4.name)
    # print(b_conv4.name)
    # print(w_conv5.name)
    # print(b_conv5.name)
    # print(w_fc1.name)
    print(b_fc1.name)
    print(W_fc2.name)
    print(b_fc2.name)
    print(tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter(FLAGS.summaries_dir,graph_def=sess.graph_def)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            _,acc=sess.run([train_step, accuracy],feed_dict={x: train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], keep_prob: 0.5})
            print('epoch %i, step %d,minibatch %i / %i, train acc %f' % (epoch, iter, minibatch_index + 1, n_train_batches,acc))
            if (iter + 1) % 100 == 0:
                valid_acc=0
                for i in range(n_test_batches):
                    acc=sess.run([accuracy],feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
                    valid_acc =valid_acc+ acc[0]
                valid_acc=valid_acc/n_test_batches
                print('                         validation acc %g' %(valid_acc ))
                if valid_acc>best_validation_acc:    
                    best_validation_acc=valid_acc
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"])
                    with tf.gfile.FastGFile('t100.pb', mode = 'wb') as f:
                        f.write(output_graph_def.SerializeToString())
                
    
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

