import tensorflow as tf
import numpy as np
import os
import pickle
import time
from weight import c2c_std,acc_w_c2f,acc_w_f2f

infile = 't401.pb'
outfile = 't501.pb'
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

def set_variable(var,w1,name):
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return tf.Variable(var,name =name)

def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name =name)
def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x,W,name):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID',name=name)
def conv2d2_same(x,W,name):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='SAME',name=name)
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
    # wcov1=sess.run('conv1/w:0',{'input_x:0':test_set_x,'label_y:0':test_set_y,'keep_prob:0':1.0})
    # bcov1 = sess.run('conv1/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    #
    # wcov13a1 = sess.run('con1_incep3a1/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov13a1 = sess.run('con1_incep3a1/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # wcov23a1 = sess.run('con2_incep3a1/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov23a1 = sess.run('con2_incep3a1/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    #
    # wcov13a2 = sess.run('con1_incep3a2/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov13a2 = sess.run('con1_incep3a2/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # wcov23a2 = sess.run('con2_incep3a2/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov23a2 = sess.run('con2_incep3a2/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    #
    # wcov13a3 = sess.run('con1_incep3a3/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov13a3 = sess.run('con1_incep3a3/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    #
    # wcov23a4 = sess.run('con2_incep3a4/w:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    # bcov23a4 = sess.run('con2_incep3a4/b:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})

    wcov2 = sess.run('conv2/w_conv2:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    bcov2 = sess.run('conv2/b_conv2:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    wcov3 = sess.run('conv3/w_conv3:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    bcov3 = sess.run('conv3/b_conv3:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})

    wfc1 = sess.run('fc1/w_fc1:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    bfc1 = sess.run('fc1/b_fc1:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    wfc2 = sess.run('fc2/w_fc2:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})
    bfc2 = sess.run('fc2/b_dc2:0', {'input_x:0': test_set_x, 'label_y:0': test_set_y, 'keep_prob:0': 1.0})

sess.close()
#################################################调整参数进行剪枝###################################################################
alpha=0.35
p=0.35
wcov2o ,bcov2o, wcov3o, out1 =c2c_std( wcov2,bcov2,wcov3,alpha,p)
print(len(out1))
alpha=0.35
p=0.35
wcov3o,bcov3o,wfc1o,out2 =acc_w_c2f(wcov3o,bcov3,wfc1,alpha,p)
print(len(out2))
alpha=0.35
p=0.35
wfc1o,bfc1o,wfc2o, out3 =acc_w_f2f(wfc1o,bfc1,wfc2,alpha,p)
print(len(out3))
bfc2o = bfc2


n_train_batches = train_set_x.shape[0]
n_test_batches  = test_set_x.shape[0]
n_train_batches = int(n_train_batches / batch_size)
n_test_batches  = int(n_test_batches  / batch_size)
print("... building the model")
#################################################重新定义新模型###################################################################
g=tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='label_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    x_images = tf.reshape(x, [-1, 28, 28, 1], name='x_tensor')
    tf.summary.image('input', x_images, 10)

    with tf.name_scope("conv2"):
        w_cov2 = set_variable(wcov2o,w1=0.0001,name = 'w_conv2')
        b_cov2 = set_variable(bcov2o,w1=0.00,name = 'b_conv2')
        h_cov2 = tf.nn.relu(conv2d2_same(x_images, w_cov2, name = 'conv2') + b_cov2,name = 'relu_conv2')

    with tf.name_scope("conv3"):
        w_cov3 = set_variable(wcov3o,w1=0.0001,name = 'w_conv3')
        b_cov3 = set_variable(bcov3o,w1=0.00,name = 'b_conv3')
        h_cov3 = tf.nn.relu(conv2d2_same(h_cov2, w_cov3, name = 'conv3') + b_cov3,name = 'relu_conv3')
        #h_pool3 = max_pool_2x2(h_cov3,name = 'maxpooling_conv3')
    with tf.name_scope("fc1"):
        h_pool2_reshape = tf.reshape(h_cov3, [-1, 7*7*len(out2)], name='cnn_fc_convert')
        w_fc1 = set_variable(wfc1o,w1=0.0001,name = 'w_fc1')
        b_fc1 = set_variable(bfc1o,w1=0.00,name = 'b_fc1')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, w_fc1) + b_fc1,name= 'relu_fc1')
    with tf.name_scope("dropout"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name= 'dropout')
    with tf.name_scope("fc2"):
        w_fc2 = set_variable(wfc2o,w1=0.0001,name = 'w_fc2')
        b_fc2 = set_variable(bfc2o,w1=0.00,name = 'b_dc2')
        y_conv = tf.nn.bias_add(tf.matmul(h_fc1_drop, w_fc2), b_fc2,name = 'y')
    with tf.name_scope("loss"):
        loss=loss(labels=y, logits=y_conv)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    print("... training")

    print(tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # summaries合并
    merged = tf.summary.merge_all()
    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter('F:/sum/train', sess.graph)

    sess=tf.Session(graph=g)
    sess.run(tf.global_variables_initializer())
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy/accuracy"])
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
            _,summary,acc=sess.run([train_step,merged, accuracy],feed_dict={x: train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size],
                y: train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], keep_prob: 0.5})
            print('epoch %i, step %d,minibatch %i/%i, train acc %g ' %
                  (epoch, iter, minibatch_index + 1, n_train_batches,acc))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            train_writer.add_run_metadata(run_metadata, 'step%03d' % iter)
            train_writer.add_summary(summary, iter)
        
        valid_acc=0
        for i in range(n_test_batches):
                acc=sess.run(accuracy,feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
                valid_acc =valid_acc+acc
        valid_acc=valid_acc/n_test_batches
        print('                         validation acc %g' %(valid_acc ))
        if valid_acc > best_validation_acc:
                best_validation_acc = valid_acc
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy/accuracy"])
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