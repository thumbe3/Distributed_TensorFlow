from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.summary_io import SummaryWriterCache

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("batch_size", 50, "batch size")
tf.app.flags.DEFINE_integer("regression_iter", 1500, "No. of weight updates")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "10.10.1.1:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "node0:2222"
    ],
    "worker" : [
        "node0:2223",
        "node1:2222",
        "node2:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
log_dir = "./logs"

if FLAGS.job_name == "ps":
 server.join()
 print("Parameter server log!!!!!")
elif FLAGS.job_name == "worker":
 g = tf.Graph()
 with g.as_default():
  x = tf.placeholder(tf.float32,[None, 784],name='input')
  y = tf.placeholder(tf.float32,[None, 10],name='label')
  W = tf.Variable(tf.zeros([784, 10]),name='Weights')
  b = tf.Variable(tf.zeros([10]),name='Bias')
  y_pred = tf.nn.softmax(tf.matmul(x, W) + b,name='softmax_output')
  loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=1))
  tf.summary.scalar("Loss", loss)    # define summary for loss for tensorborad visualization 
  optim = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1)) 
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
  tf.summary.scalar("Accuracy", accuracy)
  init = tf.initialize_all_variables() 
  summary_var = tf.summary.merge_all()

  writer = SummaryWriterCache.get(log_dir) 
  sess = tf.Session()
  sess.run(init)
  for i in range(FLAGS.regression_iter):
   data_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
   sess.run(optim, feed_dict={x:data_batch, y:label_batch})
   if i%100==0:

     summary, acc = sess.run([summary_var, accuracy],feed_dict={x:mnist.test.images,y:mnist.test.labels})
     print(acc)
     writer.add_summary(summary,i)
     #tf.summary.FileWriter("logs/MNIST_%d_%d_%s_%d" %(FLAGS.batch_size,FLAGS.regression_iter,FLAGS.deploy_mode,FLAGS.task_index),sess.graph)

  
 sess.close()
 print(acc)
