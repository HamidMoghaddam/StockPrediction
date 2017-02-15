'''
Created on 16/02/2017

@author: smas255
'''
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from matplotlib import pyplot
# Read data from csv file
def get_data(filename):    
    # Read file
    priceDataset=np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    # Scale data   
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))    
    X_dataset=scaler.fit_transform(priceDataset[...,0])
    Y_dataset=scaler.fit_transform(priceDataset[...,1])
    # Split data to training and test 
    X_train, X_test, y_train, y_test = train_test_split(
        X_dataset, Y_dataset, test_size=0.3, random_state=1234)
    return scaler,X_train, X_test, y_train, y_test
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with Tanh activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.tanh(layer_1)

    # Hidden layer with Tanh activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)   

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
if __name__ == '__main__':  
    # Create training and test 
    scaler, x_train, x_test, y_train, y_test =get_data('./data/aapl.csv')   
    # Reshape data for a network with single input and output
    x_train=np.reshape(x_train, (-1, 1))
    y_train=np.reshape(y_train, (-1, 1))
    x_test=np.reshape(x_test, (-1, 1))
    y_test=np.reshape(y_test, (-1, 1))
    # Get size of training 
    total_len = x_train.shape[0]
    # Parameters
    learning_rate = 0.01
    training_epochs = 700
    batch_size = 5
    display_step = 1
    
    # Network Parameters
    n_hidden_1 = 15 # 1st layer number of features
    n_hidden_2 = 7 # 2nd layer number of features    
    n_input = x_train.shape[1]
    n_output = 1
    
    # tf Graph input
    x = tf.placeholder("float", [None, 1])
    y = tf.placeholder("float", [None,1])
    
    # Create weights and bias vector with constant data to be similar to Azure machine learning service 
    weights_1 = np.empty([n_input, n_hidden_1],dtype=np.float32)
    weights_2 = np.empty([n_hidden_1, n_hidden_2],dtype=np.float32)
    weights_3 = np.empty([n_hidden_2, n_output],dtype=np.float32)
    weights_1.fill(0.1)
    weights_2.fill(0.1)
    weights_3.fill(0.1)
    bias_1=np.empty(n_hidden_1,dtype=np.float32)
    bias_2=np.empty(n_hidden_2,dtype=np.float32) 
    bias_3=np.empty(n_output,dtype=np.float32)     
    bias_1.fill(0)
    bias_2.fill(0)
    bias_3.fill(0)
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.constant(weights_1),dtype=tf.float32),
        'h2': tf.Variable(tf.constant(weights_2),dtype=tf.float32),        
        'out': tf.Variable(tf.constant(weights_3),dtype=tf.float32)
    }
    biases = {
        'b1': tf.Variable(tf.constant(bias_1),dtype=tf.float32),
        'b2': tf.Variable(tf.constant(bias_2),dtype=tf.float32),        
        'out': tf.Variable(tf.constant(bias_3),dtype=tf.float32)
    }
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.square(pred-y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
    
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(total_len/batch_size)
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x = x_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c,p= sess.run([optimizer, cost, pred], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch    
         
            print ("num batch:", total_batch)
    
            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
                
                print ("[*]============================")
    
        print ("Optimization Finished!")       
        # Make predictions
        y_predicted=pred.eval(feed_dict={x: x_test})
        # Invert predictions
        y_predicted=scaler.inverse_transform(y_predicted)
        y_test=scaler.inverse_transform(y_test)
        # Calculate root mean squared error
        score = math.sqrt(metrics.mean_squared_error(y_predicted, y_test))
        # Plot data
        pyplot.plot(y_predicted, color="blue")
        pyplot.plot(y_test, color="green")
        pyplot.show()
        print('RMSE: {0:f}'.format(score))