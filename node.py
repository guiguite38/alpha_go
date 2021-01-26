# TODO : load models and not models weights, so we don't need to import the package

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
print('GPU FOUND :)' if tf.config.list_physical_devices("GPU") else 'No GPU :(')
import numpy as np
import math
from Goban import Board
from Annexes.ml_go import create_model, name_to_coord, make_board
from copy import deepcopy


class node():
    """
    Class used for mcts calibrations : explorations, rollouts, rollbacks.
    Parameters:
        - father : father node
        - board : tensor 2*9*9 (??)
        - gnugo_board : gnugo version of board (??)
        - move : move from father to child
    """
    
    _MODEL_VICTORY = None
    _MODEL_PLAY_STONE = None

    def __init__(self, father, board, gnugo_board, move):
        self.nb_child_wins = 0
        self.nb_child_visit = 0
        # self.c = math.sqrt(2) # empirically the best value (?????)
        self.board = board
        self.gnugo_board = gnugo_board
        # print(f"[node.__init__] gnugo_board {gnugo_board}")
        self.A = node.get_victory(board) # s is predicted proba of victory / then sum(A_children) / visits
        self.father=father #needed for rollback
        self.children = []
        self.move = move # from father

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
        """
        print("should be : raise NotImplementedError('We're screwed. (get_player_turn)')")
        return 0        
    
    def explore_node(self):
        """Explores all children of node"""
        player_turn=self.get_player_turn()
        
        # if player_turn==1:
        #     # player is black -> need to reverse (training was only done as white)
        #     black_stones=board[1].copy()
        #     white_stones=board[0].copy()
        #     tmp_board=[black_stones, white_stones]
        # else:
        tmp_board = deepcopy(self.board)

        # check legal moves        
        child_gnugo_board=deepcopy(self.gnugo_board)
        legal_moves = self.gnugo_board.legal_moves()
                
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)
        #print(f"[node.explore_node] legal_moves {legal_moves}")

        # explore legal moves
        for move in legal_moves:
            # print(f"[node.explore_node] gnugoboard {self.gnugo_board}")
                
            [x,y] = name_to_coord(Board.flat_to_name(move))            
            board_legal_moves[x,y] = 1

            # make move on local board
            child_board = deepcopy(tmp_board)
            child_board[x,y] = 1

            # push on gnugo_board
            child_gnugo_board.push(move)
            # print(f"[node.explore_node] pushed move : {Board.coord_to_name((x,y))}")
            # print(f"[node.explore_node] pushed move as int : {move}")

            child = node(self,child_board,child_gnugo_board,move)
            self.children.append(child)
            self.nb_child_visit+=1

    
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
        child_gnugo_board=deepcopy(self.gnugo_board)
        legal_moves = self.gnugo_board.legal_moves()
        
        # print(f"[node.rollout] gnugoboard {self.gnugo_board}")
        # print(f"[node.rollout] child_gnugoboard {child_gnugo_board}")
        # print(f"[node.rollout] legal_moves {legal_moves}")
        
        # convert playable coords to board
        board_legal_moves = np.zeros((9, 9), dtype=bool)
        
        for move in legal_moves:
            x,y = name_to_coord(Board.flat_to_name(move))
            board_legal_moves[x,y] = 1

        # combine predictions with legal moves
        legal_predict = np.reshape(prediction[0],(9,9))*board_legal_moves

        if(len(legal_moves)==1):
            # player must pass
            child_board = deepcopy(self.board)
            child_gnugo_board.push(-1)
            child = node(self,child_board,child_gnugo_board,-1)
            self.children.append(child)
        else:
            # take best and push
            index_best = np.argmax(legal_predict)
            x = index_best % 9 # !!should be board_size
            y = index_best // 9 # !!should be board_size
            child_board = deepcopy(self.board)

            # print(f"[node.rollout] child_board {child_board[x,y][0]}")

            child_board[x,y][0] = 1

            tmp_move=Board.name_to_flat(Board.coord_to_name((x,y)))
            # print(f"[node.rollout] move {tmp_move}")
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
                # TODO : handle draw
                raise(NotImplementedError("can't handle draw"))
            return result == "0-1" # black wins if True
    



    