# TODO : load models and not models weights, so we don't need to import the package

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print('GPU FOUND :)' if tf.config.list_physical_devices("GPU") else 'No GPU :(')
import numpy as np
import math
from Goban import Board
from Annexes.ml_go import create_model, name_to_coord, make_board


class node():
    """
    DOCUMENTATION (!!asifwe'devergetthere)
    """
    
    _MODEL_VICTORY = None
    _MODEL_PLAY_STONE = None

    def __init__(self, father, board, gnugo_board):
        self.nb_child_wins = 0
        self.nb_child_visit = 0
        self.c = math.sqrt(2) # empirically the best value (?????)
        self.board = board
        self.gnugo_board = gnugo_board
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

        # check legal moves
        child_gnugo_board=self.gnugo_board.copy()
        legal_moves = self.gnugo_board.legal_move()
        
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)
        for move in legal_moves:
            x,y = name_to_coord(move)
            board_legal_moves[x,y] = 1
        
        # TODO: here it would be wise to use gnugoban -> forbidden moves ! mama mia...
        for i,line in enumerate(tmp_board[player_turn]):
            for j,_ in enumerate(line):
                if board_legal_moves[i,j] == 1: # explore legal_move
                    # make move on local board
                    child_board = tmp_board.copy()
                    child_board[i,j] = 1

                    # push on gnugo_board
                    child_gnugo_board.push(Board.coord_to_name((i,j)))

                    child = node(self,child_board,child_gnugo_board)
                    self.children.append(child)
    
    def rollback(self, victory):
        """
        parameters :
            victory is the boolean output from rollout
        """
        # here we need to update every parameter using the rollout ouput 
        self.nb_child_wins += victory
        self.nb_child_visit += 1
        self.A = np.sum([child.A for child in self.children]) / self.nb_child_visit # proba of victory is sum(A_children) / visits

    def rollout(self):
        """Play according to model_stone until the end
        returns result"""
        
        # make prediction for every slot
        prediction = node.get_play_stone(self.board)

        # check legal moves
        child_gnugo_board=self.gnugo_board.copy()
        legal_moves = self.gnugo_board.legal_move()
        
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)
        
        for move in legal_moves:
            x,y = name_to_coord(move)
            board_legal_moves[x,y] = 1        

        # combine predictions with legal moves
        legal_predict = prediction[0]*board_legal_moves

        # take best and push
        index_best = np.argmax(legal_predict)
        x = index_best % 9 # !!should be board_size
        y = index_best // 9 # !!should be board_size
        child_board = self.board.copy()
        child_board[0][x,y] = 1
        child_gnugo_board.push(Board.coord_to_name((x,y)))

        child = node(self,child_board,child_gnugo_board)
        self.children.append(child)
        
        # continue rollouts while not game_over
        game_over=self.gnugo_board.is_game_over()        
        if not game_over:
            return not child.rollout()
        else:
            result = self.gnugo_board.result()
            if result == "1/2-1/2":
                # TODO : handle draw
                raise(NotImplementedError("can't handle draw"))
            return result == "0-1" # black wins if True
    



    