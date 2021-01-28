# -*- coding: utf-8 -*-
""" This is the file you have to modify for the tournament.
Your default AI player must be called by this module, in the myPlayer class.
"""

import time
import Goban
from random import choice
from playerInterface import *
from mcts import mcts


class myPlayer(PlayerInterface):
    """
    Player relying on mcts predictions.
    """

    def __init__(self):
        self._goban_board = Goban.Board()
        self._mycolor = None


    def getPlayerName(self):
        return "Hopefully-smarter-than-random Player"


    def getPlayerMove(self):
        if self._goban_board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
            
        black_indexes = [i for i,x in enumerate(self._goban_board._board) if x==1]
        white_indexes = [i for i,x in enumerate(self._goban_board._board) if x==2]

        black_stones = [Goban.Board.flat_to_name(i) for i in black_indexes]
        white_stones = [Goban.Board.flat_to_name(i) for i in white_indexes]

        my_mcts = mcts()
        move = my_mcts.chose_best_action(black_stones = black_stones, white_stones=white_stones,gnugo_board=self._goban_board) 

        self._goban_board.push(move)

        return Goban.Board.flat_to_name(move)


    def playOpponentMove(self, move):
        print("Opponent played ", move)
        self._goban_board.push(Goban.Board.name_to_flat(move))


    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)


    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")

