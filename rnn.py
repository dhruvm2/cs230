#NOTE: THE MAJORITY OF THIS CODE COMES FROM THE CODE FOR THE TENSORFLOW TUTORIAL IN COURSE 2. IT HAS BEEN INVALUABLE IN THE CREATION OF THIS PROJECT.

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

import pandas as pd

stream_data = pd.read_csv('stream_data.csv', quotechar='"', skipinitialspace=True).values
rank_data = pd.read_csv('rank_data.csv', quotechar='"', skipinitialspace=True).values

n=stream_data.shape[0]
d=stream_data.shape[1]

n=stream_data.shape[0]
d=stream_data.shape[1]

RNN_X=[]
RNN_Y=[]

for i in range(n):
    print(i)
    j = 3
    while( d >= j + 32):
        if(np.sum([np.isnan(x) for x in stream_data[i][j:(j+32)]]) == 0):
            subX = []
            for k in range(30):
                subsubX = []
                subsubX.append(stream_data[i][j+1+k])
                subsubX.append(rank_data[i][j+1+k])
                subsubX.append(1.0 if stream_data[i][j+k+1]>stream_data[i][j+k] else 0.0)
                subsubX.append(1.0 if rank_data[i][j+k+1]>rank_data[i][j+k] else 0.0)
                day = np.zeros(6)
                if((j+1+k)%7 < 6):
                    day[(j+1+k)%7] = 1.0
                subsubX.extend(day)
                subX.append(subsubX)
            if(all((all(isinstance(item, float) for item in sub) for sub in subX))):
                    if(np.sum(np.isnan(subX))==0):
                        RNN_X.append(subX)
                        RNN_Y.append([stream_data[i][j+31]])
            #RNN_X.append(subX)
            #RNN_Y.append([stream_data[i][j+31]])
            j += 32
        else:
            j += 1
    
RNN_X = np.array(RNN_X)
RNN_Y = np.array(RNN_Y)


s = np.arange(RNN_X.shape[0])
np.random.shuffle(s)

RNN_X = RNN_X[s]
RNN_Y = RNN_Y[s]

m = RNN_X.shape[0]

cutoff =  int(m*4/5)

train_input = RNN_X[:cutoff]
train_output = RNN_Y[:cutoff]
test_input = RNN_X[cutoff:]
test_output = RNN_Y[cutoff:]

def create_placeholders(n_x, n_y, time_len, num_var):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape=(None,time_len, num_var))
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    ### END CODE HERE ###
    
    return X, Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters(hidden_dimension):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [1, 124]
                        b1 : [1, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    cell = tf.nn.rnn_cell.LSTMCell(hidden_dimension,state_is_tuple=True)
    W1 = tf.get_variable("W1", [hidden_dimension,1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [1,1], initializer = tf.constant_initializer(0.0))
    ### END CODE HERE ###

    parameters = {"cell": cell,
                  "W1": W1,
                  "b1": b1}
    
    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z1 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    cell = parameters['cell']
    W1 = parameters['W1']
    b1 = parameters['b1']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    val, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    Z1 = tf.add(tf.matmul(last,W1),b1)                                             
    ### END CODE HERE ###
    
    return Z1

# GRADED FUNCTION: compute_cost 

def compute_cost(Z1, Y,reg_constant,parameters):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean((Z1/Y - 1)**2)
    W1 = parameters['W1']
    b1 = parameters['b1']
    cost = cost + (tf.reduce_sum(W1 * W1) )*reg_constant
    ### END CODE HERE ###
    
    return cost

def random_mini_batches(X_train, Y_train, minibatch_size, seed):
    m = X_train.shape[0]
    
    start = 0
    end = minibatch_size
    minibatches = []
    while(end < m):
        minibatches.append((X_train[start:end],Y_train[start:end]))
        start += minibatch_size
        end += minibatch_size
    return minibatches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,reg_constant = 0.01,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    #(n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    #n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    hidden_dimension = 10
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(0, 0,30,hidden_dimension)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(hidden_dimension)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z1 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z1, Y,reg_constant,parameters)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.initialize_all_variables()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Train Cost after epoch %i: %f" % (epoch, epoch_cost))
                _ , test_cost = sess.run([optimizer, cost], feed_dict={X: X_test, Y: Y_test})
                print("Test Cost after epoch %i: %f" % (epoch, test_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        return parameters
    
parameters = model(train_input, train_output, test_input, test_output,learning_rate = 200.0,reg_constant = 0.000000001,minibatch_size = 100,num_epochs = 300)
