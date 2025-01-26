import sklearn.metrics as metrics

from rnn import rnn
from data_process import data_processing
import matplotlib.pyplot as plt
import numpy as np

# Constants and global variables
NUM_HIDDEN_LAYERS = 7
NUM_OF_FEATURES = 3
EPOCHS = 100
LEARNING_RATE = 0.01


# Purpose: Plots the predicted and actual closing prices
# Inputs:
#   prediction_values - Predicted closing prices
#   Y_training_data - Training labels
#   Y_testing_data - Test labels
#   autoscaler - Scaler object for inverse transforming data
def plot_prediction(prediction_values, Y_training_data, Y_testing_data, data_scaler):
    # Reformat the training values back to dollars
    original_Y_train = data_scaler.inverse_transform(Y_training_data.reshape(-1, 1)).flatten()
    original_Y_test = data_scaler.inverse_transform(Y_testing_data.reshape(-1, 1)).flatten()
    train_length = len(original_Y_train)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot training set
    train_line, = ax.plot(original_Y_train, label='Train Set Close Prices', color='orange')

    # Plot testing set
    test_line, = ax.plot(range(train_length, train_length + len(prediction_values[0][0])),
                         original_Y_test[:len(prediction_values[0][0])], label='Actual Close Price', color='red')

    # Plot predictions
    prediction_lines = []
    for prediction, num_layers in prediction_values:
        scaled_predictions = data_scaler.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
        pred_line, = ax.plot(range(train_length, train_length + len(prediction)),
                             scaled_predictions, label=f'Predicted Close Price Layers={num_layers}')
        prediction_lines.append(pred_line)

    plt.xlabel("Instance Number")
    plt.ylabel("Predicted Closing Value")
    legend = plt.legend()

    # Make legend clickable
    lines = [train_line, test_line] + prediction_lines
    legend_lines = legend.legend_handles

    for leg_line, plot_line in zip(legend_lines, lines):
        leg_line.set_linewidth(0) 
        leg_line.set_marker("s")  
        leg_line.set_markersize(10) 
        leg_line.set_picker(True)  
        leg_line.set_pickradius(5) 

    def on_pick(event):
        legend_item = event.artist
        idx = legend_lines.index(legend_item)
        plot_line = lines[idx] 
        visible = not plot_line.get_visible()
        plot_line.set_visible(visible)
        legend_item.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle() 

    fig.canvas.mpl_connect("pick_event", on_pick)


    for i in range(3, 6):
        plot_line = lines[i] 
        visible = not plot_line.get_visible() 
        plot_line.set_visible(visible)
        legend.legend_handles[i].set_alpha(1.0 if visible else 0.2) 


    plt.show()
    plt.show()


# Purpose: Tests the trained RNN model and calculates RMSE and R^2 for each prediction
# Inputs:
#   generated_rnn - Trained RNN model
#   num_hidden_layers - Number of hidden layers in the RNN
#   num_of_features - Number of features in the data
#   X_test - Test features
#   Y_test - Test labels
# Returns:
#   metrics.mean_squared_error - RMSE value for the predictions
#   metrics.r2_score - R^2 values for the predictions
#   point_prediction - List of predicted values
def test_rnn(generated_rnn, num_hidden_layers, num_of_features, X_test, Y_test):
    point_prediction = []
    actual_values = []

    # Test the RNN
    for i in range(len(X_test) - num_hidden_layers):
        long, _ = generated_rnn.perform_prediction(X_test[i:i + num_hidden_layers], num_of_features)
        actual_value = Y_test[i + num_hidden_layers]

        point_prediction.append(long)
        actual_values.append(actual_value)

    return (metrics.mean_squared_error(Y_test[num_hidden_layers:], point_prediction),
            metrics.r2_score(Y_test[num_hidden_layers:], point_prediction),
            point_prediction)


# Purpose: Trains the RNN model with the given configurations.
# Inputs:
#   num_hidden_layers - Number of hidden layers in the RNN.
#   num_features - Number of features in the data.
#   learning_rate - Learning rate for training.
# Returns:
#   trained_rnn - Trained RNN model.
#   X_train_vals - Training features.
#   X_test_vals - Test features.
#   Y_train_vals - Training labels.
#   Y_test_vals - Test labels.
#   scaler - Scaler object for data transformation.
def train_rnn(num_hidden_layers, num_features, learning_rate):
    X_train_vals, X_test_vals, Y_train_vals, Y_test_vals, scaler = data_processing()

    X_train_vals = np.array(X_train_vals)
    Y_train_vals = np.array(Y_train_vals)

    trained_rnn = rnn(num_hidden_layers, num_features, learning_rate)

    times = 0
    i = 0
    # Train the RNN
    while times < EPOCHS:
        long, short = trained_rnn.perform_prediction(X_train_vals[i:i + num_hidden_layers], num_features)
        actual = Y_train_vals[i + num_hidden_layers]
        cost = trained_rnn.calculate_cost(actual, short)
        trained_rnn.back_propagation(Y_train_vals, num_hidden_layers)

        if i + num_hidden_layers >= len(X_train_vals):
            i = 0

        times += 1

    return trained_rnn, X_train_vals, np.array(X_test_vals), Y_train_vals, np.array(Y_test_vals), scaler


# Purpose: Main function to execute the training, testing, and evaluation of the RNN model.
def main():
    my_rnn, X_train, X_test, Y_train, Y_test, autoscaler = train_rnn(NUM_HIDDEN_LAYERS, NUM_OF_FEATURES, LEARNING_RATE)
    print("TESTING MODEL")
    print("Results...\nTrain/Test Split = 70:30\nSize of dataset =", len(X_test) + len(X_train))

    print()
    print("-------------------------------------")
    print("RNN:\nNumber of layers = {}\nError Function = {}\nLearning Rate = {}\n"
          "Number of Epochs = {}".format(NUM_HIDDEN_LAYERS, "RMSE/R^2", LEARNING_RATE, EPOCHS))

    # Test with global variables
    rmse_val, r2_val, predictions = test_rnn(my_rnn, NUM_HIDDEN_LAYERS, NUM_OF_FEATURES, X_train, Y_train)
    print(f'Training RMSE: {rmse_val = :.4f}')
    print(f'Training R^2: {r2_val = :.4f}')
    rmse_val, r2_val, predictions = test_rnn(my_rnn, NUM_HIDDEN_LAYERS, NUM_OF_FEATURES, X_test, Y_test)
    print(f'Test RMSE: {rmse_val = :.4f}')
    print(f'Test R^2 : {r2_val = :.4f}')

    # Test with set values
    print()
    print("Number of layers = 3\nError Function RMSE/R^2\nLearning rate = 3\nNumber of Epochs = 100")
    generated_rnn, _, _, _, _, _ = train_rnn(3, 3, 0.01)
    rmse_val, r2_val, predictions3 = test_rnn(generated_rnn, 3, 3, X_test, Y_test)
    print(f'Test RMSE: {rmse_val = :.4f}')
    print(f'Test R^2 : {r2_val = :.4f}')

    # Test with set values
    print()
    print("Number of layers = 5\nError Function RMSE/R^2\nLearning rate = 3\nNumber of Epochs = 100")
    generated_rnn, _, _, _, _, _ = train_rnn(5, 3, 0.01)
    rmse_val, r2_val, predictions5 = test_rnn(generated_rnn, 5, 3, X_test, Y_test)
    print(f'Test RMSE: {rmse_val = :.4f}')
    print(f'Test R^2 : {r2_val = :.4f}')

    # Test with set values 
    print()
    print("Number of layers = 10\nError Function RMSE/R^2\nLearning rate = 3\nNumber of Epochs = 100")
    generated_rnn, _, _, _, _, _ = train_rnn(10, 3, 0.01)
    rmse_val, r2_val, predictions10 = test_rnn(generated_rnn, 10, 3, X_test, Y_test)
    print(f'Test RMSE: {rmse_val = :.4f}')
    print(f'Test R^2 : {r2_val = :.4f}')

    plot_prediction([(predictions, NUM_HIDDEN_LAYERS), 
                     (predictions3, 3), 
                     (predictions5, 5), 
                     (predictions10, 10)], 
                     Y_train, Y_test, autoscaler)


if __name__ == "__main__":
    main()
