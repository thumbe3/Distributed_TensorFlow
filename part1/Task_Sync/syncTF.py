#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.summary_io import SummaryWriterCache
import time

# use this function to retry downloading mnist if it fails (Taken from stack_overflow)
def download_mnist_retry(seed=0, max_num_retries=20):
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets("MNIST_data", one_hot=True,
                                             seed=seed)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")


# to print statistics like mean, stddev etc of weights onto tensorboard
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

#mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# define the command line flags that can be sent

tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_integer('regression_iter', 1500,
                            'No. of weight updates')
tf.app.flags.DEFINE_integer('task_index', 0,
                            'Index of task with in the job.')
tf.app.flags.DEFINE_string('job_name', 'worker', 'either worker or ps')
tf.app.flags.DEFINE_string('deploy_mode', 'single',
                           'either single or cluster')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({'worker': ['localhost:2222'
        ]})

clusterSpec_cluster = tf.train.ClusterSpec({'ps': ['10.10.1.1:2222'],
        'worker': ['10.10.1.1:2223', '10.10.1.2:2222']})

clusterSpec_cluster2 = tf.train.ClusterSpec({'ps': ['10.10.1.1:2222'],
        'worker': ['10.10.1.1:2223', '10.10.1.2:2222', '10.10.1.3:2222'
        ]})

clusterSpec = {'single': clusterSpec_single,
               'cluster': clusterSpec_cluster,
               'cluster2': clusterSpec_cluster2}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)
log_dir = './logs_sync'
converging_loss = 0.35
mnist = download_mnist_retry(seed=FLAGS.task_index)
if FLAGS.job_name == 'ps':
    server.join()
    print('Parameter server log!!!!!')
elif FLAGS.job_name == 'worker':
    start_time = time.time()

    # use device replica to replicate among all workers
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'
                    % FLAGS.task_index, cluster=clusterinfo)):


        # define the placeholder for input and label
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        y = tf.placeholder(tf.float32, [None, 10], name='label')

        #get mean std_dev and other statistics of W and b as it gets updated
        with tf.name_scope("Weights"):
          W = tf.Variable(tf.zeros([784, 10]), name='Weights')
          variable_summaries(W)
        with tf.name_scope("Bias"):
          b = tf.Variable(tf.zeros([10]), name='Bias')
          variable_summaries(b)

        #find the loss corresponding to the weights and bias
        y_pred = tf.nn.softmax(tf.matmul(x, W) + b,
                               name='softmax_output')
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred),
                              reduction_indices=1))
        tf.summary.scalar('Loss', loss)  # define summary for loss for tensorborad visualization


        # seperate subgraph of model training in tensorboard for better visualization
        with tf.name_scope('Model_Training'):

            # defining apply_gradients as a global step between workers
            # using a syncreplica optimzer to sync the model updates 
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=clusterinfo.num_tasks('worker'),total_num_replicas=clusterinfo.num_tasks('worker'))
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                    global_step=global_step)

        # find the accuracy (used to plot test_accuracy) and summary writer initialization for tensorboard
        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)
        init = tf.initialize_all_variables()	
        summary_var = tf.summary.merge_all()
        writer = SummaryWriterCache.get(log_dir)


        # create a monitored training session obect and pass a hook to it indicating the chief being worker 0
        config = tf.ConfigProto(device_filters=['/job:ps',
                                '/job:worker/task:%d'
                                % FLAGS.task_index])
        is_chief = FLAGS.task_index==0
        hook=optimizer.make_session_run_hook(is_chief, num_tokens=0)
        saver = tf.train.Saver()
        mts = tf.train.MonitoredTrainingSession(master=server.target,
                is_chief= is_chief and FLAGS.job_name
                == 'worker', config=config,
                hooks=[hook])

        iterations = 0
        with mts as sess:

            # run till test loss is more than the convergin_loss( loss that is desired)
            while True:
                (data_batch, label_batch) = \
                    mnist.train.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: data_batch,
                         y: label_batch})

                # See the test loss and test accuracy every 100 iterations
                if iterations % 100 == 0:
                    (summary, acc,l) = sess.run([summary_var, accuracy,loss],feed_dict={x: mnist.test.images,y: mnist.test.labels})
                    writer.add_summary(summary, iterations)
                    if l<converging_loss:
                        print("Converged in %d iterations"%iterations)
                        break
                iterations+=1	
            print("Accuracy obtained %f"%acc) 
           
            time_taken = time.time()-start_time
            print("Total time take by worker %d is %s"%(FLAGS.task_index, time_taken))




			
