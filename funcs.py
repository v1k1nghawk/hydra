
import sys
from os import path
from inspect import stack
import numpy as np
import keras
from keras import layers
from classes.datapoint import Datapoint
from classes.dataset import Dataset



def generate_model(model_dirpath: str, x_leftmost: int = 1, x_rightmost: int = 102) -> keras.Model:
    '''Generate (create, train, validate, test and save) a new model based on a generated temporary dataset.

    :param model_dirpath: the path where the model's files will be saved
    :param x_leftmost: the leftest bound of the generated temporary dataset's X
    :param x_rightmost: the rightest bound of the generated temporary dataset's X

    :returns: trained multi-layer perceptron NN model

    '''
    ##############################
    # Generate temporary dataset
    ##############################
    num_of_features = x_rightmost - x_leftmost - 1
    num_of_datapoints=10000
    ds = Dataset(x_leftmost, x_rightmost, num_of_datapoints)
    ds.generate()

    #######
    # Prepare vectorized subsets for the model
    #######
    print("Vectorizing subsets...", end="", flush=True)

    # Get subsets (as input (X) and output (Y) variables) in matrix form from generated dataset
    X_train, y_train = ds.vectorized_X_Y_train
    X_validation, y_validation = ds.vectorized_X_Y_valid
    X_test, y_test = ds.vectorized_X_Y_test

    print("...OK\n")

    ##############################
    # Create the model (multi-layer perceptron)
    ##############################
    print("Model creating...", end="", flush=True)

    #######
    # Model specification
    #######
    # Input - datapoint's feature vector
    mlp = keras.Sequential([
        layers.Input(shape=(num_of_features,), dtype=np.float64, name="input_layer"),
        layers.Dense(num_of_features*2, activation='relu', name="hidden_layer1"),
        layers.Dropout(0.2), # Prevent overfitting
        layers.Dense(num_of_features*2, activation='relu', name="hidden_layer2"),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', name="output_layer")
    ])
    # Output - probability of the given feature vector belonging to a 2-spiked Gaussian Mixture class


    #######
    # Model compilation
    #######
    mlp.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("...OK\n")


    #######
    # Train and validate the model
    #######
    print("Model training...")

    training_results = mlp.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                               epochs=100, batch_size=8, verbose=2, shuffle=True)

    print('TRAIN {accuracy: %.2f}, {loss: %.2f}'
          % (training_results.history['accuracy'][-1]*100, training_results.history['loss'][-1]))
    print('VALID {accuracy: %.2f}, {loss: %.2f}\n'
          % (training_results.history['val_accuracy'][-1]*100, training_results.history['val_loss'][-1]))


    #######
    # Test the model
    #######
    print("Model evaluation...")

    test_results = mlp.evaluate(X_test, y_test, verbose=0)
    print("TEST {%s: %.2f}, {loss: %.2f}\n" % (mlp.metrics_names[1], test_results[1]*100, test_results[0]))


    #######
    # Save model
    #######
    print("Saving model...", end="", flush=True)

    try:
        # Save model's computational graph with weights
        mlp.save(model_dirpath)

    except IOError as e:
        print("...IOError: <" + str(e) + ">\n")
    else:
        print("...OK\n")

    return mlp


def load_model(model_dirpath: str) -> keras.Model:
    '''Load existing pre-trained model from a storage.

    :param model_dirpath: path to the saved model

    :returns: NN model

    '''
    print("Model loading...", end="", flush=True)

    try:
        # Load model's architecture with weights
        model = keras.models.load_model(model_dirpath)

    except IOError as e:
        print("...IOError: <" + str(e) + ">\n")
        return np.nan
    except ValueError as e:
        print("...ValueError: <" + str(e) + ">\n")
        return np.nan
    else:
        print("...OK\n")
        return model


def main(argv, arc) -> int:
    '''Load or generate the model to predict the label for a given datapoint.

    :param lower_bound: the leftest bound of the given datapoint's X
    :param upper_bound: the rightest bound of the given datapoint's X

    :returns: status (0 if successful)

    :raises TypeError: incorrect arguments' type, model getting error
    :raises ValueError: incorrect number of arguments, model<->datapoint incompatibility

    '''
    ##############################
    # Arguments' handling
    ##############################
    # Get overall bounds of slots (bins, segments) in each datapoint
    if arc == 3:
        try:
            X_lower_bound=int(argv[1])
            X_upper_bound=int(argv[2])
        except ValueError:
            raise TypeError(stack()[0][3] + ' args: program arguments must be integer type')
        else:
            if X_lower_bound <= 0:
                raise ValueError(stack()[0][3] + ' args: lower_bound <= 0')
            elif (X_upper_bound - X_lower_bound) <= 100:
                raise ValueError(stack()[0][3] + ' args: upper_bound - lower_bound <= 100')
    else:
        raise ValueError('Correct way to call this program is: ' + argv[0] + ' \"lower_bound\" \"upper_bound\"')


    ##############################
    # Get the model
    ##############################
    duospikes_finder = np.nan
    model_dir_name="model_" + str(X_lower_bound) + "_" + str(X_upper_bound)
    if path.exists(model_dir_name) == True:
        duospikes_finder = load_model(model_dir_name)
    else:
        duospikes_finder = generate_model(model_dir_name, X_lower_bound, X_upper_bound)

    if isinstance(duospikes_finder, keras.Model) != True:
        raise TypeError(stack()[0][3] + ': Given model is not keras.Model type')

    print("Model's architecture...")
    duospikes_finder.summary()


    ##############################
    # Predict label for new datapoint
    ##############################
    print("Prediction with the model...")

    # Get new datapoint
    try:
        # Load external datapoint
        dp_pred = Datapoint().load("fresh_datapoint", False)
    except IOError:
        print("Brand new point creating...")
        dp_pred = Datapoint(X_lower_bound, X_upper_bound)
        dp_pred.generate()

    # Check model<->datapoint compatibility
    if duospikes_finder.get_layer('hidden_layer1').input_shape[1] != dp_pred.feature_vector_length:
        raise ValueError(stack()[0][3]
                         + ': model\'s input dimensions=<' + str(duospikes_finder.get_layer('hidden_layer1').input_shape[1])
                         + '> incompatible with datapoint\'s fv dimensions=<' + str(dp_pred.feature_vector_length) + '>')

    # Transform datapoint to model-understandable form
    X_pred = dp_pred.feature_vector
    print("X_pred", X_pred)
    print("y_pred ACTUAL", dp_pred.actual_label)

    # Run inference using the model
    y_pred_estim = duospikes_finder.predict(np.expand_dims(X_pred,axis=0))
    y_pred_estim = np.where(y_pred_estim > 0.5, 1, 0)
    print("y_pred ESTIMATED", bool(y_pred_estim[0,0]))

    dp_pred.predicted_label = bool(y_pred_estim[0,0])
    dp_pred.show()
    dp_pred.save("fresh_datapoint")

    return 0

