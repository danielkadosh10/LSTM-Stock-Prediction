# Running the Code
To run the code and get the results and graph, simply run the main.py file. It will test the model with our preset best 
hyperparameters and then generate a graph of the predictions vs the actual stock values. It will also plot 
three more predictions using different hyperparamteters.

# Load Your Own Data
Change the data in the data.csv, it has to be in the following format.
'Date,Open,High,Low,Close,Adj Close,Volume'

# Files
This project is split into different files. The files and their use are listed below
- main: Trains and tests the recurrent neural network (RNN). It includes functions to run error tests, plot predictions, and test the RNN model. It evaluates the performance of the RNN using root mean squared error (RMSE) and R^2 error and plots the predicted and actual stock prices.
- data_process: Performs data processing on stock market data. It reads a CSV file containing stock data, selects a subset of days, scales the features and target label using StandardScaler, and splits the data into training and testing sets, returning the processed data along with the scaler object.
- activation_funcs: Implements the sigmoid and tanh activation functions.
- gate_stage: Defines methods to initialize and update weights and biases.
- gates: Defines several gate classes (forget_gate, input_gate, cell_gate, output_gate) that utilize the gate_stage class from gate_stage.py for LSTM operations.
- lstm: Defines an LSTM (Long Short-Term Memory) class, which is composed of forget, input, cell, and output gates, each with its own set of weights and biases.
- rnn: Defines a recurrent neural network (RNN) class.