# TODO : load models and not models weights, so we don't need to import the package

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print('GPU FOUND :)' if tf.config.list_physical_devices("GPU") else 'No GPU :(')
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb

def create_model(**kwargs):
    '''
    Input is 9*9*2 for black and white positions
    Output is 1 with probability of player victory
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

    # print(model.summary())
    return model


class node():
    """
    DOCUMENTATION (!!asifwe'devergetthere)
    """
    
    _MODEL_VICTORY = None
    _MODEL_PLAY_STONE = None

    def __init__(self, father, board):
        self.nb_child_wins = 0
        self.nb_child_visit = 0
        self.c = sqrt(2) # empirically the best value
        self.board = board
        self.A = node.get_victory(board) # s is predicted proba of victory / then sum(A_children) / visits
        self.father=father #needed for rollback
        self.children = []


    @staticmethod
    def get_victory(board):
        """Returns proba of victory for node.board"""
        if node._MODEL_VICTORY == None:
                node._MODEL_VICTORY=create_model(prior=False)
                node._MODEL_VICTORY.load_weights("./models/model_victory/")
        return node._MODEL_VICTORY.predict(np.array([board]))

    @staticmethod
    def get_play_stone(board):
        """Returns a board with proba of playing for each node.board cell"""
        if node._MODEL_PLAY_STONE == None:
                node._MODEL_PLAY_STONE=create_model(prior=False)
                node._MODEL_PLAY_STONE.load_weights("./models/model_prior/")
        return node._MODEL_PLAY_STONE.predict(np.array([board]))

    def get_player_turn(self):
        """
        The tensor layer (0 or 1) where next stone will be played
        """
        raise NotImplementedError("We're screwed. (get_player_turn)")
    
    def explore_node(self):
        """Explores all children of node"""
        player_turn=self.get_player_turn()
        
        # if player_turn==1:
        #     # player is black -> need to reverse (training was only done as white)
        #     black_stones=board[1].copy()
        #     white_stones=board[0].copy()
        #     tmp_board=[black_stones, white_stones]
        # else:
        tmp_board = self.board.copy()
        
        # TODO: here it would be wise to use gnugoban -> forbidden moves ! mama mia...
        for i,line in enumerate(tmp_board[player_turn]):
            for j,cell in enumerate(line):
                if cell == 0: # should be 'if legal_move'
                    child_board = tmp_board.copy()
                    child_board[i,j] = 1
                    self.children.append(node(self,child_board))

    def rollout(self):
        """Play according to model_stone until the end
        returns result"""
        
        #TODO -> play recursively + get final score
        prediction = get_play_stone(self.board)
        index_best = np.argmax(prediction[0])
        x = index_best % 9 # !!should be board_size
        y = index_best // 9 # !!should be board_size
        child_board = self.board.copy()
        child_board[x,y] = 1
        child = node(self,child_board)
        self.children.append(child)
        
        # TODO define end -> connect with gnugoban
        game_over=False
        if not game_over:
            return child.rollout()
        else:
            # TODO : implement return child.has_won?
            raise NotImplementedError("We're screwed. (whether child has one)")
    



    