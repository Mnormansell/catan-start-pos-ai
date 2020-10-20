"""
RUN A GAME OF CATAN with AI
"""

import catan.board
from collections import namedtuple
from utils.catan_graph import CatanGraph


# Build a board, turn tuple, create a problem, solve?

class Game:
    CatanState = collections.namedtuple('CatanState', 'name age gender')
    
    def init(player = 1, n_player = 3, board = None, player_strategy = None, enemy_agent = None):
        self.gamestate = 
        pass

    def turn(self):
        """
        Execute a turn: settlement + road
        """

    def run(self):
        """
        Execute overall game logic
        """
        pass