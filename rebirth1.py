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

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      '/root/ipythoncmd/prun','pr1.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def load_data(dataset_path):
    def gene_label(x):
        l=len(x)
        labelout=np.zeros([l,10], dtype='int32')
        for i in range(l):
            labelout[i][x[i]]=1
        return labelout
    dic={}
    fo = open(os.path.join(dataset_path, 'data_batch_1'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    ##[number,rgb,width,height],1024r+1024g+1024b
    train=dic['data']
    train_label=gene_label(dic['labels'])
    fo = open(os.path.join(dataset_path, 'data_batch_2'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    train=np.vstack((train,dic['data']))
    train_label=np.vstack((train_label,gene_label(dic['labels'])))
    fo = open(os.path.join(dataset_path, 'data_batch_3'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    train=np.vstack((train,dic['data']))
    train_label=np.vstack((train_label,gene_label(dic['labels'])))
    fo = open(os.path.join(dataset_path, 'data_batch_4'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    train=np.vstack((train,dic['data']))
    train_label=np.vstack((train_label,gene_label(dic['labels'])))
    fo = open(os.path.join(dataset_path, 'data_batch_5'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    train=np.vstack((train,dic['data']))
    train_label=np.vstack((train_label,gene_label(dic['labels'])))
    fo = open(os.path.join(dataset_path, 'test_batch'), 'rb')
    dic.update(pickle.load(fo, encoding='ISO-8859-1'))
    test=dic['data']
    test_label=gene_label(dic['labels'])
    fo.close()
    rval = [(train, train_label), (test,test_label)]
    return rval

def set_variable(var,w1):
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return tf.Variable(var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0.0,stddev=0.001)
    return tf.Variable(initial)

def conv2d2(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2 ,2, 1], padding='SAME')

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1,1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def mean_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

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
n_epochs=200
datasets = load_data(r'/root/ipythoncmd/quanti/cifar-10-batches-py')
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[1]

with tf.Session() as sess:
    create_graph()
    output_tensor  = sess.graph.get_tensor_by_name('Variable:0')
    output_tensor1 = sess.graph.get_tensor_by_name('Variable_1:0')
    output_tensor2 = sess.graph.get_tensor_by_name('Variable_2:0')
    output_tensor3 = sess.graph.get_tensor_by_name('Variable_3:0')
    output_tensor4 = sess.graph.get_tensor_by_name('Variable_4:0')
    output_tensor5 = sess.graph.get_tensor_by_name('Variable_5:0')
    output_tensor6 = sess.graph.get_tensor_by_name('Variable_6:0')
    output_tensor7 = sess.graph.get_tensor_by_name('Variable_7:0')
    output_tensor8 = sess.graph.get_tensor_by_name('Variable_8:0')
    output_tensor9 = sess.graph.get_tensor_by_name('Variable_9:0')
    output_tensor10 = sess.graph.get_tensor_by_name('Variable_10:0')
    output_tensor11 = sess.graph.get_tensor_by_name('Variable_11:0')
    output_tensor12 = sess.graph.get_tensor_by_name('Variable_12:0')
    output_tensor13 = sess.graph.get_tensor_by_name('Variable_13:0')

    a=sess.run(output_tensor,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a1=sess.run(output_tensor1,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a2=sess.run(output_tensor2,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a3=sess.run(output_tensor3,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a4=sess.run(output_tensor4,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a5=sess.run(output_tensor5,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a6=sess.run(output_tensor6,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a7=sess.run(output_tensor7,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a8=sess.run(output_tensor8,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a9=sess.run(output_tensor9,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a10=sess.run(output_tensor10,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a11=sess.run(output_tensor11,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a12=sess.run(output_tensor12,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
    a13=sess.run(output_tensor13,{'Placeholder:0':test_set_x,'Placeholder_1:0':test_set_y,'Placeholder_2:0':1.0})
sess.close()

# 计算各数据集的batch个数
n_train_batches = train_set_x.shape[0]
n_test_batches  = test_set_x.shape[0]
n_train_batches = int(n_train_batches / batch_size)
n_test_batches  = int(n_test_batches  / batch_size)
print("... building the model")

g=tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3072])
    y = tf.placeholder(tf.int64, shape=[None,10])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]),perm=[0,2,3,1])
    w_conv1 = set_variable(a,w1=0.001) 
    b_conv1 = set_variable(a1,w1=0)
    h_pool1 = tf.nn.relu(tf.nn.bias_add(conv2d2(x_image, w_conv1), b_conv1))

    w_conv2 = variable_weight_loss(shape=[5, 5, 16,64],stddev=5e-2,w1=0.001)
    b_conv2 = bias_variable([64])
    h_pool2 = tf.nn.relu(tf.nn.bias_add(conv2d2(h_pool1, w_conv2) , b_conv2))

    w_conv3 = set_variable(a4,w1=0.001)
    b_conv3 = set_variable(a5,w1=0.0)
    h_pool3 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2, w_conv3) , b_conv3)))

    w_conv4 = set_variable(a6,w1=0.001)
    b_conv4 = set_variable(a7,w1=0.0)
    h_pool4 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool3, w_conv4) , b_conv4)))

    w_conv5 = set_variable(a8,w1=0.001)
    b_conv5 = set_variable(a9,w1=0.0)
    h_pool5 = mean_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv2d(h_pool4, w_conv5) , b_conv5)))

    h_fc1_drop = tf.nn.dropout(h_pool5, keep_prob)
    w_fc1 = set_variable(a10,w1=0.004)
    b_fc1 = set_variable(a11,w1=0.0)
    y_fc1 = tf.nn.bias_add(tf.matmul(tf.reshape(h_fc1_drop, [-1, 1 * 1 * 83]), w_fc1), b_fc1)

    h_fc2_drop = tf.nn.dropout(y_fc1, keep_prob)
    w_fc2 = set_variable(a12,w1=0.04)
    b_fc2 = set_variable(a13,w1=0.0)
    y_conv = tf.nn.bias_add(tf.matmul(h_fc2_drop, w_fc2) , b_fc2)

    loss=loss(labels=y,logits=y_conv)
    var1 = tf.trainable_variables()[2:4]
    var2 = tf.trainable_variables()[0:2]
    var3 = tf.trainable_variables()[4:]
    train_step1= tf.train.GradientDescentOptimizer(0.01).minimize(loss,var_list=var1)
    train_step2= tf.train.GradientDescentOptimizer(0.001).minimize(loss,var_list=var2)
    train_step3= tf.train.GradientDescentOptimizer(0.001).minimize(loss,var_list=var3)
    train_step = tf.group(train_step1, train_step2,train_step3)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    tf.summary.scalar('accuracy', accuracy)

    sess=tf.Session(graph=g)
    sess.run(tf.global_variables_initializer())

    best_validation_acc = 0
    epoch = 0
    done_looping = False

    print("... training")
    print(x.name)
    print(y.get_shape())
    print(y_conv.get_shape())
    print(accuracy.name)
    print(w_conv1.name)
    print(b_conv1.name)
    print(w_conv2.name)
    print(b_conv2.name)
    print(w_conv3.name)
    print(b_conv3.name)
    print(w_conv4.name)
    print(b_conv4.name)
    print(w_conv5.name)
    print(b_conv5.name)
    print(w_fc1.name)
    print(b_fc1.name)
    print(w_fc2.name)
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
                # compute zero-one loss on validation set
                valid_acc=0
                for i in range(n_test_batches):
                    acc=sess.run([accuracy],feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
                    valid_acc =valid_acc+ acc[0]
                valid_acc=valid_acc/n_test_batches
                print('                         validation acc %g' %(valid_acc ))
                if valid_acc > best_validation_acc:
                    best_validation_acc=valid_acc
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["accuracy"])
                    #print(output_graph_def)
                    with tf.gfile.FastGFile('/root/ipythoncmd/prun/pr2.pb', mode ='wb') as f:
                        f.write(output_graph_def.SerializeToString())
    print('Optimization complete')
    test_acc=0
    start_time=time.time()
    valid_acc=0
    print(start_time)
    for i in range(n_test_batches):
        valid_acc =valid_acc+ accuracy.eval(feed_dict={x: test_set_x[i*batch_size:(i+1)*batch_size], y: test_set_y[i*batch_size:(i+1)*batch_size], keep_prob:1})
    end_time=time.time()
    test_acc=valid_acc/n_test_batches
    print("test accuracy %g" % test_acc)
