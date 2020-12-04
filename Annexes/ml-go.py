import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb


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


## DONNEES EN ENTREE ET SORTIE DU MODELE


def position_predict(black_stones, white_stones):

    # ... Votre tambouille interne pour placer les pierres comme il faut dans votre structure de données
    # et appeler votre modèle Keras (typiquement avec model.predict())
    prediction = None  # model.predict(...) # A REMPLIR CORRECTEMENT

    return prediction


# Ainsi, pour le rendu, en admettant que newdata soit le fichier json contenant les nouvelles données que
# l'on vous donnera 24h avant la fin, vous pourrez construire le fichier resultat ainsi


def create_result_file(newdata):
    """ Exemple de méthode permettant de générer le fichier de resultats demandés. """
    resultat = [position_predict(d["black_stones"], d["white_stones"]) for d in newdata]
    with open("my_predictions.txt", "w") as f:
        for p in resultat:
            f.write(str(p) + "\n")


######################################################
######################################################

# First steps: transform all the data into numpy arrays to feed your neural network


def name_to_coord(s):
    """
    Prends en entrée une unique coord sous forme de chaine de caratères A2
    Retourne une paire d'entiers représentant la coordonnée
    """
    assert s != "PASS"
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

    # print(sample["white_stones"])
    # print(sample["black_stones"])

    for white_stone in sample["white_stones"]:
        x, y = name_to_coord(white_stone)
        board[x, y, 0] = True

    for black_stone in sample["black_stones"]:
        x, y = name_to_coord(black_stone)
        board[x, y, 1] = True
    return board


# white stones at the following position ['C8', 'E7', 'H7', 'G3']
# black stones at the following position ['B7', 'C6', 'E5', 'C3']


# if name == "__main__":
# Test of board making
# rval = np.random.randint(0,)
# board = make_board(data[10])
# print(board.shape)
# print(board[1, 6, 1])

# print(board.T[0])
# print(board.T[1])

# print("We have", len(data),"examples")
# # summary_of_example(data,10)
# # Par exemple, nous pourrons appeler votre prédiction ainsi

# print("Prediction this sample:")
# summary_of_example(data, 10)
# print()
# prediction = position_predict(data[10]["black_stones"], data[10]["white_stones"])
# print("You predicted", prediction, "and the actual target was", data[10]["black_wins"]/data[10]["rollouts"])


######################################################
######################################################

# Second steps: build your neural network and train it

# on fait le rollout
# pour l'améliorer on veut jouer des coups probables pour accélérer le roll out
# donc on veut la proba de jouer un coup à partir d'un plateau donné

# fonction utilitaire pour visualisation de l'entraînement (cours bpesquet)
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
def count_freedom():
    pass

# on veut make un board avec extraction préalable du dernier qui sera considéré comme un label
def create_dataset_prior(data):
    """
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


data = get_raw_data_go()
x, y = create_dataset_prior(data)

print(len(x))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

print(len(x_train),len(x_test),len(y_train),len(y_test))

# (9,9,2) -> conv2D -> conv2D -> sigmoid(Dense) -> (9,9)


####################################
######### MODEL DEFINITION #########
####################################
def create_conv_model():
    '''
    Input is 9*9*2 for black and white positions
    Output is 9*9=81 with probability distribution for the next stone
    '''
    model = Sequential()

    model.add(
        Conv2D(filters=128, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=64, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=32, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(
        Conv2D(filters=16, kernel_size=(3,3), padding='same', data_format="channels_last", input_shape=(9, 9, 2))
    )
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

    model.add(Dense(1, activation='relu'))

    print(model.summary())
    return model

def custom_loss(y_actual,y_pred):
    '''
    loss = sum(x_dist²+y_dist²) sur le top 5
    shapes are 81*1
    '''
    # TO FIX : AttributeError: 'Tensor' object has no attribute 'argsort'
    top5 = y_pred.argsort()[:5]
    # transform en 9*9 coord pour avoir la distance
    coords = [[arg%9,(arg-arg%9)/9] for arg in top5]

    actual_arg = y_actual.argmax()
    actual_coord = [actual_arg%9,(actual_arg-actual_arg%9)/9]
    
    custom_loss=sum((actual_coord[0]-coord[0])**2 + (actual_coord[1]-coord[1])**2 for coord in coords)
    return custom_loss

model=create_conv_model()
model.compile(optimizer='adam',loss='mse',metrics=["mae", "mse"])    
print("***** Compilation is DONE *****")

history = model.fit(x_train,y_train,epochs=10,verbose=1,batch_size=128,validation_split=0.1)
print("***** Model fit is DONE *****")

plot_loss_acc(history)
_, test_acc = model.evaluate(x_test,y_test,verbose=0)
print("Evaluation : testing accuracy = ", test_acc)

# print(x_test[:1],type(x_test[:1]))
# pred = np.array([res.argsort[:5] for res in model.predict(x_test)])

pred=[]
for res in model.predict(x_test):
    pred.append(res.argsort()[:5])
np.array(pred)

success = np.sum(np.array([1 for i in range(len(pred)) if np.argmax(y_test[i]) in pred[i]]))
print("Accuracy : ",success/len(x_test))


# print([np.argmax(model.predict([x_test[i]])) for i in range(len(y_test))])
# print([np.argmax([y_test[i]]) for i in range(len(y_test))])