# -*- coding: utf-8 -*-
""" Different tools to define, train and interact with models. """

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import tensorflow.keras.backend as kb
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # Prepared for calculus infrastructure by CATIE
print('GPU FOUND :)' if tf.config.list_physical_devices("GPU") else 'No GPU :(')


## LECTURE DES DONNEES

def get_raw_data_go():
    """ Returns the set of samples from the local file or download it if it does not exists"""
    import gzip, os.path
    import json

    raw_samples_file = "samples-9x9.json.gz"

    if not os.path.isfile(raw_samples_file):
        print("File", raw_samples_file, "not found, I am downloading it...", end="")
        import urllib.request

        urllib.request.urlretrieve(
            "https://www.labri.fr/perso/lsimon/ia-inge2/samples-9x9.json.gz",
            "samples-9x9.json.gz",
        )
        print(" Done")

    with gzip.open("samples-9x9.json.gz") as fz:
        data = json.loads(fz.read().decode("utf-8"))
    return data


## COMPREHENSION DES DONNEES

def summary_of_example(data, sample_nb):
    """ Gives you some insights about a sample number"""
    sample = data[sample_nb]
    print("Sample", sample_nb)
    print()
    print("Données brutes en format JSON:", sample)
    print()
    print("The sample was obtained after", sample["depth"], "moves")
    print("The successive moves were", sample["list_of_moves"])
    print(
        "After these moves and all the captures, there was black stones at the following position",
        sample["black_stones"],
    )
    print(
        "After these moves and all the captures, there was white stones at the following position",
        sample["white_stones"],
    )
    print(
        "Number of rollouts (gnugo games played against itself from this position):",
        sample["rollouts"],
    )
    print(
        "Over these",
        sample["rollouts"],
        "games, black won",
        sample["black_wins"],
        "times with",
        sample["black_points"],
        "total points over all this winning games",
    )
    print(
        "Over these",
        sample["rollouts"],
        "games, white won",
        sample["white_wins"],
        "times with",
        sample["white_points"],
        "total points over all this winning games",
    )


## TRANSFORM DATA TO FEED OUR NEURAL NETWORK


def name_to_coord(s):
    """
    Prends en entrée une unique coord sous forme de chaine de caratères A2
    Retourne une paire d'entiers représentant la coordonnée
    """
    if s == "PASS":
        return [-1,-1]
    
    indexLetters = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "J": 8,
    }

    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return [col, lin]


def make_board(sample):
    """
    Creates representation of go board
    2 grids -> 1 for white stones, 1 for black stones = (9,9,2) matrix
    """
    board = np.zeros((9, 9, 2), dtype=bool)

    for white_stone in sample["white_stones"]:
        x, y = name_to_coord(white_stone)
        board[x, y, 0] = True

    for black_stone in sample["black_stones"]:
        x, y = name_to_coord(black_stone)
        board[x, y, 1] = True
    return board


## BUILD AND TRAIN NEURAL NETWORK

# tool for visualisation training (bpesquet.fr/mlkatas)
def plot_loss_acc(history):
    """Plot training and (optionally) validation loss and accuracy"""

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, '.--', label='Training loss')
    final_loss = loss[-1]
    title = 'Training loss: {:.4f}'.format(final_loss)
    plt.ylabel('Loss')
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'o-', label='Validation loss')
        final_val_loss = val_loss[-1]
        title += ', Validation loss: {:.4f}'.format(final_val_loss)
    plt.title(title)
    plt.legend()

    acc = history.history['accuracy']

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, '.--', label='Training acc')
    final_acc = acc[-1]
    title = 'Training accuracy: {:.2f}%'.format(final_acc * 100)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if 'val_accuracy' in history.history:
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, val_acc, 'o-', label='Validation acc')
        final_val_acc = val_acc[-1]
        title += ', Validation accuracy: {:.2f}%'.format(final_val_acc * 100)
    plt.title(title)
    plt.legend()


def create_dataset_victory(data):
    """
    Preprocess data to train the victory network
    """
    input_data = []
    output_data = []
    for sample in data:
        if "PASS" not in sample["list_of_moves"]:
            output_name = sample["list_of_moves"][-1]
            input_sample = {}
            input_sample["black_stones"] = sample["black_stones"][:-1]
            input_sample["white_stones"] = sample["white_stones"]

            input_data.append(make_board(input_sample))
            output_data.append(sample["black_wins"]/sample["rollouts"])
        else:
            pass
    return np.array(input_data), np.array(output_data)


def create_dataset_prior(data):
    """
    Preprocess data to train the rollout network
    Takes list of moves as an input and returns [data,label]
    with label = last move
    """
    input_data = []
    output_data = []
    for sample in data:
        if "PASS" not in sample["list_of_moves"]:
            output_name = sample["list_of_moves"][-1]
            input_sample = {}

            if len(sample["black_stones"]) > len(sample["white_stones"]):
                input_sample["black_stones"] = sample["black_stones"][:-1]
                input_sample["white_stones"] = sample["white_stones"]
            else:
                input_sample["black_stones"] = sample["black_stones"]
                input_sample["white_stones"] = sample["white_stones"][:-1]

            x,y = name_to_coord(output_name)
            output_board = np.zeros((9,9))
            output_board[x,y] = 1
            
            output_data.append(output_board.reshape(81))
            input_data.append(make_board(input_sample))
        else:
            pass
    return np.array(input_data), np.array(output_data)


## MODEL DEFINITION


def create_model(**kwargs):
    '''
    Input is 9*9*2 for black and white positions
    Output is :
    - 1 with probability of player victory
    - or 81 with probability of victory for each square (enable stone placement)
    13 convolution layers  into 3 dense
    '''
    prior=kwargs.get("prior",True)

    model = Sequential()

    model.add(
        Conv2D(filters=128, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if prior:
        # Stone prediction model
        model.add(Dense(81, activation='sigmoid'))
    else:        
        # Victory prediction model
        model.add(Dense(1, activation='relu'))
    return model



def train_model(x_train,y_train,model):
    model.compile(optimizer='adam',loss="mse",metrics=["mae", "mse","accuracy"], run_eagerly=True)    
    print("***** Compilation is DONE *****")

    history = model.fit(x_train,y_train,epochs=10,verbose=1,batch_size=128,validation_split=0.1)
    print("***** Model fit is DONE *****")
    return history


def test_model(x_test, y_test, history, model):
    plot_loss_acc(history)

    print(f"[model.evaluate] : {model.evaluate(x_test,y_test,verbose=0)}")
    loss, mae, mse, test_acc = model.evaluate(x_test,y_test,verbose=0)
    print("Evaluation : testing accuracy = ", test_acc)

    # Eval model on test data
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("loss, mae, mse, test_acc :", results)
    y_pred = model.predict(x_test)
    tf.keras.losses.MAE(
        y_test, y_pred
    )

    # TODO use custom loss function (BELOW) to determine wether results are acceptable or not
    success = np.sum(np.array([1 for i in range(len(y_pred)) if np.max(y_test[i]) in y_pred[i]]))
    print("Accuracy : ",success/len(x_test))


def custom_loss_obsolete(y_actual,y_pred):
    '''
    loss = sum(x_dist²+y_dist²) sur le top 5
    shapes are 81*1
    y_actual is 81*zeroes with 1 on the coord where the stone should be placed
    y_pred is distribution of probabilities
    '''
    np_y_pred=np.array(y_pred)
    np_y_actual=np.array(y_actual)
    top5 = np.argsort(np_y_pred)[:5]
    print(f"[custom_loss] top5 : {top5}")
    
    # transform en 9*9 coord pour avoir la distance
    coords = [[arg%9,(arg-arg%9)/9] for arg in top5]
    print(f"[custom_loss] coords : {coords}")

    actual_arg = np.argmax(np_y_actual)
    actual_coord = [actual_arg%9,(actual_arg-actual_arg%9)/9]
    print(f"[custom_loss] actual_coord : {actual_coord}")

    custom_loss=sum((actual_coord[0]-coord[0])**2 + (actual_coord[1]-coord[1])**2 for coord in coords)
    print(f"[custom_loss] computed custom_loss : {custom_loss}")
    return custom_loss


if __name__ =="__main__":
    data = get_raw_data_go()
    x_prior, y_prior = create_dataset_prior(data)
    x_victory, y_victory = create_dataset_victory(data)

    x_train_p,x_test_p,y_train_p,y_test_p = train_test_split(x_prior,y_prior,test_size=0.1)
    x_train_v,x_test_v,y_train_v,y_test_v = train_test_split(x_victory,y_victory,test_size=0.1)

    ## UNCOMMENT TO TEST ##
    # print(len(x_train),len(x_test),len(y_train),len(y_test))

    ## (9,9,2) -> conv2D -> conv2D -> sigmoid(Dense) -> (9,9)

    # Model creation and training
    # model_prior=create_model(prior=True)
    # history_prior = train_model(x_train_p, y_train_p, model_prior)
    # test_model(x_test_p, y_test_p, history_prior, model_prior)
    # model_prior.save_weights("./models/model_prior/")

    # model_victory=create_model(prior=False)
    # history_victory = train_model(x_victory, y_victory,model_victory)
    # test_model(x_test_v, y_test_v, history_victory, model_victory)
    # model_victory.save_weights("./models/model_victory/")
    
    # Model Loading
    # model_victory=create_model(prior=False)
    # model_victory.load_weights("./models/model_victory/")
    # prob_victory = model_victory.predict(np.array([x_test_v[0]]))

    model_prior=create_model(prior=True)
    model_prior.load_weights("./models/model_prior/")
    prob_prior = model_prior.predict(np.array([x_test_p[0]]))
    print(prob_prior.shape)
    



