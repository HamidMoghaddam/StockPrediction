'''
Created on 16/02/2017

@author: smas255
'''
import time
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
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
def build_model():
    model = Sequential()
    # Define layers   
    layers = [1, 15, 7,1]
    # Create weights and bias vector with constant data to be similar to Azure machine learning service 
    weightsOne = np.empty([layers[0],layers[1]],dtype=float)
    biasOne=np.empty(layers[1],dtype=float)
    weightsOne.fill(0.1)
    biasOne.fill(0)
    
    weightsTwo = np.empty([layers[1],layers[2]],dtype=float)
    biasTwo=np.empty(layers[2],dtype=float)
    weightsTwo.fill(0.1)
    biasTwo.fill(0)
    
    weightsThree = np.empty([layers[2],layers[3]],dtype=float)
    biasThree=np.empty(layers[3],dtype=float)
    weightsThree.fill(0.1)
    biasThree.fill(0)
    # Hidden layer with Tanh activation
    model.add(Dense(
            input_dim=layers[0],
            output_dim=layers[1], activation='tanh',weights=[weightsOne,biasOne]))  
    # Hidden layer with Tanh activation  
    model.add(Dense(layers[2],activation='tanh',weights=[weightsTwo,biasTwo]))    
    # Output layer with linear activation
    model.add(Dense(output_dim=layers[3], activation='linear',weights=[weightsThree,biasThree]))    
    # Gradient Descent Optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0, nesterov=True)
    # Create model
    model.compile(loss='mean_squared_error', optimizer=sgd)   
    return model
if __name__ == '__main__':    
    global_start_time = time.time()
    # Parameters
    learning_rate = 0.01
    training_epochs = 700
    batch_size = 5
    display_step = 1    
    # Create training and test 
    scaler, X_train, X_test, y_train, y_test =get_data('./data/aapl.csv')
    model = build_model()    
    try:
        model.fit(X_train, y_train,batch_size=batch_size, nb_epoch=training_epochs) 
        # Make predictions
        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)
        # Invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([y_train])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([y_test])
        # Calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))        
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        # Plot data
        pyplot.plot(testPredict[:,0], color="blue")
        pyplot.plot(testY[0], color="green")
        pyplot.show()
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)  
          
    pass