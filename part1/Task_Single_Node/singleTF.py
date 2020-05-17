from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.summary_io import SummaryWriterCache
import time

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
log_dir = "./logs" # tensorboard loggging directory
converging_loss = 0.35  # loss criterion to stop the worker

if FLAGS.job_name == "ps":
 server.join()
 print("Parameter server log!!!!!")
elif FLAGS.job_name == "worker":
 g = tf.Graph()
 start_time = time.time()
 with g.as_default():
  # define the placeholder for input and label
  x = tf.placeholder(tf.float32,[None, 784],name='input')
  y = tf.placeholder(tf.float32,[None, 10],name='label')

  #define variables for bias and weights
  W = tf.Variable(tf.zeros([784, 10]),name='Weights')
  b = tf.Variable(tf.zeros([10]),name='Bias')
  
  #find the loss corresponding to the weights and bias
  y_pred = tf.nn.softmax(tf.matmul(x, W) + b,name='softmax_output')
  loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=1))
  tf.summary.scalar("Loss", loss)    # define summary for loss for tensorborad visualization 

  # define a optimizer to minimize loss
  optim = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  # find the accuracy (used to plot test_accuracy)
  correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1)) 
  accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
  tf.summary.scalar("Accuracy", accuracy) #define summary for accuracy for tensorboard visualization

  #initialize variables
  init = tf.initialize_all_variables()
 
  # merge all the summary variables
  summary_var = tf.summary.merge_all()
  # create a summary writer object to log in tensorboard
  writer = SummaryWriterCache.get(log_dir) 
  sess = tf.Session()
  sess.run(init)
  
  # run till test loss is more than the convergin_loss( loss that is desired)
  iterations = 0
  while True:
      data_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
      sess.run(optim, feed_dict={x:data_batch, y:label_batch})

      # See the test loss and test accuracy every 100 iterations
      if iterations%100==0:
           summary, acc,l = sess.run([summary_var, accuracy, loss],feed_dict={x:mnist.test.images,y:mnist.test.labels})
           writer.add_summary(summary,iterations)
           if l<converging_loss:
               print("Converged in %d iterations"%iterations)
               break
      iterations = iterations + 1

  print("Accuracy obtained %f"%acc)
  time_taken = time.time()-start_time
  print("Total time take by worker %d is %s"%(FLAGS.task_index, time_taken))
  
 sess.close()
