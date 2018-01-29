import tensorflow as tf
import numpy as np
import os
import pickle
import time
from weight import c2c_std,acc_w_c2f,acc_w_f2f

infile = 'first.pb'
outfile = 't100.pb'
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

def set_variable(var,w1):
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return tf.Variable(var)


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

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

batch_size=100
n_epochs=5
mnist = load_data()
train_set_x  = mnist.train.images
train_set_y = mnist.train.labels
test_set_x = mnist.test.images
test_set_y = mnist.test.labels
#################################################根据名字提取变量值###################################################################
with tf.Session() as sess:
    create_graph()
    #output_tensor  = sess.graph.get_tensor_by_name('Variable:0')
    output_tensor1 = sess.graph.get_tensor_by_name('Variable_1:0')
    output_tensor2 = sess.graph.get_tensor_by_name('Variable_2:0')
    output_tensor3 = sess.graph.get_tensor_by_name('Variable_3:0')
    output_tensor4 = sess.graph.get_tensor_by_name('Variable_4:0')
    output_tensor5 = sess.graph.get_tensor_by_name('Variable_5:0')
    output_tensor6 = sess.graph.get_tensor_by_name('Variable_6:0')
    output_tensor7 = sess.graph.get_tensor_by_name('Variable_7:0')

    wcov1=sess.run('Variable:0',{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    bcov1=sess.run(output_tensor1,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    wcov2=sess.run(output_tensor2,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    bcov2=sess.run(output_tensor3,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    wfc1=sess.run(output_tensor4,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    bfc1=sess.run(output_tensor5,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    wfc2=sess.run(output_tensor6,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    bfc2=sess.run(output_tensor7,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})


sess.close()
#################################################调整参数进行剪枝###################################################################
alpha=0.45
p=0.45
b , b1, b2, out1 =c2c_std( wcov1,bcov1,wcov2,alpha,p)
print(len(out1))
alpha=0.35
p=0.35
b2, b3, b4, out2 =acc_w_c2f(b2,bcov2,wfc1,alpha,p)
print(len(out2))
alpha=0.35
p=0.35
b4, b5, b6, out3 =acc_w_f2f(b4,bfc1,wfc2,alpha,p)
print(len(out3))
b7 = bfc2


n_train_batches = train_set_x.shape[0]
n_test_batches  = test_set_x.shape[0]
n_train_batches = int(n_train_batches / batch_size)
n_test_batches  = int(n_test_batches  / batch_size)
print("... building the model")
#################################################重新定义新模型###################################################################
g=tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_images = tf.reshape(x, [-1, 28, 28, 1])
    print(x_images.name)
    # W_cov1 = weight_variable([5, 5, 1, 20])
    # print(W_cov1.name)
    # b_cov1 = bias_variable([20])
    # print(b_cov1.name)
    W_cov1 = set_variable(b, w1=0.001)
    b_cov1 = set_variable(b1, w1=0.0)
    h_cov1 = tf.nn.relu(conv2d(x_images, W_cov1) + b_cov1)
    h_pool1 = max_pool_2x2(h_cov1)

    # W_cov2 = weight_variable([5, 5, 20, 50])
    # print(W_cov2.name)
    # b_cov2 = bias_variable([50])
    # print(b_cov2.name)
    w_cov2 = set_variable(b2, w1=0.001)
    b_cov2 = set_variable(b3, w1=0.0)
    h_cov2 = tf.nn.relu(conv2d(h_pool1, w_cov2) + b_cov2)
    h_pool2 = max_pool_2x2(h_cov2)

    # W_fc1 = weight_variable([4 * 4 * 50, 500])
    # print(W_fc1.name)
    # b_fc1 = bias_variable([500])
    # print(b_fc1.name)
    w_fc1 = set_variable(b4, w1=0.004)
    b_fc1 = set_variable(b5, w1=0.0)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * len(out2)])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # W_fc2 = weight_variable([500, 10])
    # print(W_fc2.name)
    # b_fc2 = bias_variable([10])
    # print(b_fc2.name)
    w_fc2 = set_variable(b6, w1=0.004)
    b_fc2 = set_variable(b7, w1=0.0)

    # y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

    loss=loss(labels=y,logits=y_conv)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    sess=tf.Session(graph=g)
    sess.run(tf.global_variables_initializer())
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"])
    with tf.gfile.FastGFile('current.pb', mode ='wb') as f:
        f.write(output_graph_def.SerializeToString())
    
    best_validation_acc = 0
    epoch = 0
    done_looping = False
#################################################重新训练模型###################################################################
    print("... training")
    summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter(logdir = 'F:\\from github\\cut_weight_cnn\\graph', graph_def=sess.graph_def)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            _,acc=sess.run([train_step, accuracy],feed_dict={x: train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], keep_prob: 0.5})
            print('epoch %i, step %d,minibatch %i/%i, train acc %g ' %
                  (epoch, iter, minibatch_index + 1, n_train_batches,acc))
        
        valid_acc=0
        for i in range(n_test_batches):
                acc=sess.run(accuracy,feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
                valid_acc =valid_acc+acc
        valid_acc=valid_acc/n_test_batches
        print('                         validation acc %g' %(valid_acc ))
        if valid_acc > best_validation_acc:
                best_validation_acc = valid_acc
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"])
                with tf.gfile.FastGFile(outfile, mode ='wb') as f:
                    f.write(output_graph_def.SerializeToString())
 
start_time=time.time()
valid_acc=0
print(start_time)
for i in range(n_test_batches):
    valid_acc =valid_acc+ sess.run(accuracy,feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
end_time=time.time()
test_acc=valid_acc/n_test_batches
print("test accuracy %g" % test_acc)
print((end_time - start_time)*1000/60)