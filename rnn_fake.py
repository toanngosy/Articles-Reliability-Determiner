import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
from sklearn import preprocessing
from keras.preprocessing.text import one_hot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

num_input = 50      # 
timesteps = 1000  # 
n_dict = 50000#
num_hidden = 16    #
num_classes = 1     # 

learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 50


def get_glove():
  f = open("data/glove.6B.50d.txt",'r', encoding = 'utf8')
  model = {}
  for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
  return model

def to_glovevector(train_data, dict):
  nan = dict['.']
  
  trainX = []
  
  for doc in train_data:
    words = doc.split(" ")
    n_words = len(words)
    #print(n_words)
    temp = []
    for i in range(timesteps):
      if (i < n_words) or ():
        word = words[i]
      else: 
        word = "."
      temp.append(dict.get(word, nan))
    trainX.append(temp)
    
  return np.asarray(trainX)
  
def to_tfidfvector(train_data):
  tfidf_vect = TfidfVectorizer(max_features=n_dict)
  trainX = tfidf_vect.fit_transform(train_data)
  return trainX
  
def to_encodedlabel(train_label):
  #trainY = [[1-l, 0+l] for l in train_label]
  trainY = [l for l in train_label]
  return trainY
  
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return data_shuffle, np.asarray(labels_shuffle)

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=0.6)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
  
def rnn_run(train_data,train_label):
  dict = get_glove()
  
  trainX = train_data
  trainY = to_encodedlabel(train_label)
  
  # tf Graph input
  X = tf.placeholder("float", [None, timesteps, num_input])
  Y = tf.placeholder("float", [None, num_classes])
  
  # Define weights
  weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
  }
  biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
  }
  
  logits = RNN(X, weights, biases)
  prediction = tf.nn.sigmoid(logits)
  
  
  # Define loss and optimizer
  loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
  #global_step = tf.Variable(0, trainable=False)
  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_op)

  # Evaluate model (with test logits, for dropout to be disabled)
  #correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
  delta = tf.abs((Y - prediction))
  p5 = tf.constant(0.5)
  correct_pred = tf.cast(tf.less(delta, p5), tf.int32)

  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()

  # Start training
  with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(batch_size, trainX, trainY)
        
        batch_x = to_glovevector(batch_x, dict)
        
        #print(batch_x.shape)
        
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_y = batch_y.reshape((batch_size, num_classes))
        #print(batch_x.shape)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))  
  
  