from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import json
import os
#disable the tqdm progress bar that displays estimated remaining time (to avoid additional overhead)
tfds.disable_progress_bar() 

#tf flags to accept the input arguments with this file
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('task_index', 0,
                            'Index of task with in the job.')
tf.app.flags.DEFINE_integer('epochs',50,'number of epochs')
tf.app.flags.DEFINE_integer('buffer',10000,'shuffle buffer size')
tf.app.flags.DEFINE_integer('deploy_mode', 1,'1, 2 or 3')
tf.app.flags.DEFINE_integer('steps_per_epoch',1000,'steps per epoch when we use dataset.repeat()')
FLAGS = tf.app.flags.FLAGS

#set this verbosity to print the model performance after every epoch
tf.logging.set_verbosity(tf.logging.DEBUG)

BUFFER_SIZE = FLAGS.buffer
BATCH_SIZE = FLAGS.batch_size

# Following function does preprocessing of the input images
def scale(image,label):
 image = tf.cast(image,tf.float32)
 image /= 255 #Noramlize image data with max RGB code
 return image, label

#prepare batched train and test tensorflow datasets
datasets, info = tfds.load(name='mnist',with_info=True,as_supervised=True)
train_datasets_unbatched = datasets['train'].map(scale).cache().repeat().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)
test_datasets_unbatched = datasets['test'].map(scale).repeat()

#Building and compiling Keras model
def build_model():
 model = tf.keras.Sequential()
 model.add(layers.Conv2D(6,(3,3),strides=(1,1),activation='relu',input_shape=(28,28,1)))
 model.add(layers.AveragePooling2D(pool_size=(2,2)))
 model.add(layers.Activation('relu'))
 model.add(layers.Conv2D(16,(4,4),strides=(1,1),activation='relu'))
 model.add(layers.AveragePooling2D(pool_size=(2,2)))
 model.add(layers.Activation('relu'))
 model.add(layers.Conv2D(120,(5,5),strides=(1,1),activation='relu'))
 model.add(layers.Flatten())
 model.add(layers.Dense(84))
 model.add(layers.Activation('relu'))
 model.add(layers.Dense(10))
 model.add(layers.Activation('softmax'))

 sgd = tf.keras.optimizers.SGD(lr = 0.001)
 model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=sgd,metrics=['accuracy'])
 return model

# TF_CONFIG for various deploy modes
TF_CONFIG_ONE = {'cluster':{'worker':['10.10.1.1:2223']},'task':{'index':FLAGS.task_index,'type':'worker'}}
TF_CONFIG_TWO = {'cluster':{'worker':['10.10.1.1:2223','10.10.1.2:2222']},'task':{'index':FLAGS.task_index,'type':'worker'}}
TF_CONFIG_THREE = {'cluster':{'worker':['10.10.1.1:2223','10.10.1.2:2222','10.10.1.3:2222']},'task':{'index':FLAGS.task_index,'type':'worker'}}
Config_list = [TF_CONFIG_ONE, TF_CONFIG_TWO, TF_CONFIG_THREE]

#set TF_CONFIG environment variable for the current worker
os.environ['TF_CONFIG'] = json.dumps(Config_list[FLAGS.deploy_mode - 1])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
NUM_WORKERS = (FLAGS.deploy_mode)

#increasing the batch size to GLOBAL_BATCH_SIZE based on the number of worker we deploy
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
test_datasets = test_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)

#callback to periodically checkpoint the model data in a file
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./lenet_logs/keras-ckpt')]

#build the keras model within the strategy scope to perform auto data sharding and distributed model training
with strategy.scope():
 multi_worker_model = build_model()
multi_worker_model.fit(x=train_datasets, epochs=FLAGS.epochs, steps_per_epoch=FLAGS.steps_per_epoch, callbacks=callbacks)
