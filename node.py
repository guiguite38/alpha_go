
import tensorflow as tf
import os
import numpy as np
import math
from Goban import Board
from Annexes.ml_go import create_model, name_to_coord, make_board
from copy import deepcopy
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print('GPU FOUND :)' if tf.config.list_physical_devices("GPU") else 'No GPU :(')


class node():
    """
    Class used for mcts computations : explorations, rollouts, rollbacks.
    Parameters:
        - father : father node
        - board : np.array 2*9*9
        - gnugo_board : gnugo version of board
        - move : move from father to child
    """    
    _MODEL_VICTORY = None
    _MODEL_PLAY_STONE = None


    def __init__(self, father, board, gnugo_board, move):
        self.nb_child_wins = 0
        self.nb_child_visit = 0
        self.board = board
        self.gnugo_board = gnugo_board
        self.A = node.get_victory(board) # A is predicted proba of victory, then A = sum(A_children) / visits
        self.father=father # needed for rollback
        self.children = []
        self.move = move # move id between father node and this state in [0:80]


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
                node._MODEL_PLAY_STONE=create_model(prior=True)
                node._MODEL_PLAY_STONE.load_weights("./models/model_prior/")
        return node._MODEL_PLAY_STONE.predict(np.array([board]))


    def get_player_turn(self):
        """
        The tensor layer (0 or 1) where next stone will be played
        !! Should return player ID (0 or 1) instead of 0 !!
        """
        return 0
    

    def explore_node(self):
        """Explores all children of node"""
        player_turn=self.get_player_turn()
        tmp_board = deepcopy(self.board)

        # check legal moves        
        child_gnugo_board=deepcopy(self.gnugo_board)
        legal_moves = self.gnugo_board.legal_moves()
                
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)

        # explore legal moves
        for move in legal_moves:                
            [x,y] = name_to_coord(Board.flat_to_name(move))            
            board_legal_moves[x,y] = 1

            # make move on local board
            child_board = deepcopy(tmp_board)
            child_board[x,y] = 1

            # push on gnugo_board
            child_gnugo_board.push(move)

            child = node(self,child_board,child_gnugo_board,move)
            self.children.append(child)
            self.nb_child_visit+=1

    
    def rollback(self, victory):
        """
        Updates parameters of all father nodes with regards to rollout final result (victory).
        Parameter :
            victory is the boolean output from rollout
        """
        self.nb_child_wins += victory
        self.nb_child_visit += 1
        self.A = np.sum([child.A for child in self.children]) / self.nb_child_visit # proba of victory is sum(A_children) / visits
        

    def rollout(self):
        """Play according to model_stone until the end
        returns result"""
        
        # make prediction for every slot
        prediction = node.get_play_stone(self.board)

        # check legal moves
        child_gnugo_board=deepcopy(self.gnugo_board)
        legal_moves = self.gnugo_board.legal_moves()
                
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)
        
        for move in legal_moves:
            x,y = name_to_coord(Board.flat_to_name(move))
            board_legal_moves[x,y] = 1

        # combine predictions with legal moves
        legal_predict = np.reshape(prediction[0],(9,9))*board_legal_moves

        # take best and push
        index_best = np.argmax(legal_predict)
        x = index_best % 9 # 9 should be replaced by board_size
        y = index_best // 9 # 9 should be replaced by board_size
        child_board = deepcopy(self.board)
        child_board[x,y][0] = 1

        tmp_move=Board.name_to_flat(Board.coord_to_name((x,y)))
        if tmp_move in legal_moves:
            child_gnugo_board.push(tmp_move)
            child = node(self,child_board,child_gnugo_board,tmp_move)
            self.children.append(child)
        else:
            tmp_move = np.random.choice(legal_moves)
            child_gnugo_board.push(tmp_move)
            child = node(self,child_board,child_gnugo_board,tmp_move)
            self.children.append(child)
        
        # continue rollouts while not game_over
        game_over=child_gnugo_board.is_game_over()        
        if not game_over:
            return not child.rollout()
        else:
            result = child_gnugo_board.result()
            if result == "1/2-1/2":
                return 0
            return 1 if result == "0-1" else -1 # black wins if True
    