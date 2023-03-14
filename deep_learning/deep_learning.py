import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
from tensorflow import keras
from keras.utils import pad_sequences


def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """

    # Find the average length of the training data
    avg_length = sum(len(i) for i in data["x_train"]) // len(data["x_train"])

    maxlen = avg_length  # data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # Hyperparameters
    input_length = data["x_train"].shape[1]
    input_dim = data["vocab_size"]
    hidden_dim = 64
    hidden_layers = 2
    output_dim = 1
    activation_fnn = 'relu'
    activation_rnn = ['sigmoid', 'relu']
    recurrent_activation_rnn = 'tanh'
    output_activation = 'sigmoid'
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']
    epochs = 2
    dropout = 0.2

    # Create the model and add common layer for both feedforward and recurrent network
    model = keras.Sequential()
    model.add(
        keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=hidden_dim,
            input_length=input_length
        ))

    # Build the model given model_type
    if model_type == "feedforward":
        # Flatten the output of the embedding layer to feed it into the dense layers
        model.add(keras.layers.Flatten())
        # Using only dense layers for feedforward network
        for _ in range(hidden_layers):
            model.add(
                keras.layers.Dense(
                    units=hidden_dim,
                    activation=activation_fnn
                ))
    elif model_type == "recurrent":
        # Using LSTM layer for recurrent network
        model.add(
            keras.layers.LSTM(
                units=hidden_dim,
                activation=activation_rnn[0],
                recurrent_activation=recurrent_activation_rnn,
                dropout=dropout
            ))
        for _ in range(hidden_layers - 1):
            model.add(
                keras.layers.Dense(
                    units=hidden_dim,
                    activation=activation_rnn[1]
                ))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Defining the output layer
    # Using sigmoid activation function for both feedforward and recurrent network
    # This is due to the fact that the output is binary
    model.add(
        keras.layers.Dense(
            units=output_dim,
            activation=output_activation
        ))

    # Choosing hyperparameters that optimizes the model for binary classification
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model on train data
    model.fit(data["x_train"], data["y_train"], epochs=epochs)

    # Evaluate the models accuracy on test data
    _, test_acc = model.evaluate(data["x_test"], data["y_test"])

    # Return the accuracy of the model on test data
    return test_acc


def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')


if __name__ == '__main__':
    main()
