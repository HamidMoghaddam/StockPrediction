# StockPrediction
This is a project for predicting stock price of a company (Apple) at the end of a day based on its price at the beginning of that day. The aim of this project is to show how to make a prediction in the simplest way by TensorFlow (https://www.tensorflow.org/),  Keras (https://keras.io/), Azure Machine Learning (https://azure.microsoft.com/en-us/services/machine-learning/) and Amazon Machine Learning (https://aws.amazon.com/machine-learning/) services. Thus, a deep neural network regression model with two hidden layers is used in TensorFlow, Keras and Azure Machine Learning service. In addition a regression model is used in Amazon Machine Learning service. It is worth noting that the machine learning part of Amazon console has only regression model and Amazon Machine Image should be used for deep learning.

# TensorFlow
I tried to create a model with parameters similar to default parameters of Azure Machine Learning. Therefore, constant values are used for initial weights and biases with zero momentum. The RMSE of the model for test dataset is 1.012. Click [here](/src/root/PredictByTensorFlow.py) for the model's code.

# Keras
Keras is a high-level neural networks library, written in Python and capable of running on top of either TensorFlow or Theano (https://keras.io/). The parameters and results of this model is similar to TensorFlow. The code is provided [here](/src/root/PredictByKeras.py).

# Amazon Machine Learning

![](/pictures/AML1.png?raw=true "Loading data from s3")
![](/pictures/AML2.png?raw=true "Selecting a target column")
![](/pictures/AML4.png?raw=true "Creating a default Regression model")
![](/pictures/AML7.png?raw=true "Model is trained")
![](/pictures/AML8.png?raw=true "RMSE result")

# Azure Machine Learning
![](/pictures/Azure1.png?raw=true "Create dataset")
![](/pictures/Azure2.png?raw=true "Create experiment")
![](/pictures/Azure5.png?raw=true "Create model")
![](/pictures/Azure4.png?raw=true "Create model")
![](/pictures/Azure7.png?raw=true "Run model")

# Conclusion
Azure Machine Learning service is simple and fast to user however it is not possible to set optimizer or biases easily. In addition, it seems complex models (e.g. LSTM) should be developed by Python or R modules in Azure. As a next result, developing a model for the current project in Keras was simpler than TensorFlow but TensorFlow is more customisable and provide a good understanding of the underlying maths of the model. Although, I tried to used same parameters for TensorFlow and Azure to get same results, the RMSE of TensorFlow and Keras are less than Azure (It might be happened because of Azuer's limitations that I mensioned befor). Finally, time of processing data and running model in Amazon Machine Learning service was much more than other models.
