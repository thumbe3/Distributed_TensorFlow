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
def download_mnist_retry(seed=0, max_num_retries=20):
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets("MNIST_data", one_hot=True,
                                             seed=seed)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")

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
        'worker': ['10.10.1.2:2223', '10.10.1.3:2222']})

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

mnist = download_mnist_retry(seed=FLAGS.task_index)
if FLAGS.job_name == 'ps':
    server.join()
    print('Parameter server log!!!!!')
elif FLAGS.job_name == 'worker':
    with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'
                    % FLAGS.task_index, cluster=clusterinfo)):
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        y = tf.placeholder(tf.float32, [None, 10], name='label')
        with tf.name_scope("Weights"):
          W = tf.Variable(tf.zeros([784, 10]), name='Weights')
          variable_summaries(W)
        with tf.name_scope("Bias"):
          b = tf.Variable(tf.zeros([10]), name='Bias')
          variable_summaries(b)
        y_pred = tf.nn.softmax(tf.matmul(x, W) + b,
                               name='softmax_output')
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred),
                              reduction_indices=1))
        tf.summary.scalar('Loss', loss)  # define summary for loss for tensorborad visualization

        with tf.name_scope('Model_Training'):
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=clusterinfo.num_tasks('worker'),total_num_replicas=clusterinfo.num_tasks('worker'))
            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                    global_step=global_step)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)
        init = tf.initialize_all_variables()
	
        summary_var = tf.summary.merge_all()

        config = tf.ConfigProto(device_filters=['/job:ps',
                                '/job:worker/task:%d'
                                % FLAGS.task_index])
        is_chief = FLAGS.task_index==0
        hook=optimizer.make_session_run_hook(is_chief, num_tokens=0)
        saver = tf.train.Saver()
        mts = tf.train.MonitoredTrainingSession(master=server.target,
                is_chief= is_chief and FLAGS.job_name
                == 'worker', config=config,
                hooks=[tf.train.CheckpointSaverHook(
            checkpoint_dir=log_dir,
            save_secs=None,
            save_steps=100,
            saver=saver,
            checkpoint_basename='model_to_test.ckpt',
            scaffold=None,
            listeners=None,
            ),hook])

        writer = SummaryWriterCache.get(log_dir)
        with mts as sess:
            for i in range(FLAGS.regression_iter):
                (data_batch, label_batch) = \
                    mnist.train.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: data_batch,
                         y: label_batch})
                if i % 100 == 0:
                    if FLAGS.task_index == 0:
                        (summary, acc) = sess.run([summary_var,
                                accuracy],
                                feed_dict={x: mnist.test.images,
                                y: mnist.test.labels})
                        writer.add_summary(summary, i)
                        print(acc)

#summary_dir=log_dir
			
