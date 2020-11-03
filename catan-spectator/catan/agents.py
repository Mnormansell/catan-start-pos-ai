"""
File that stores collection of agents, i.e random agent, 
"""
from catan.states import GameStatePreGamePlacingPiece
from catan.pieces import Piece, PieceType
import hexgrid
import random


class Agent(object):
    """
    An abstract class for other agents to inherit from
    """

    def __init__(self, game, player):
        self.game = game
        self.player = player

    # Solve at that state
    def solve(self, gamestate):
        raise NotImplemented()


class RandomAgent(Agent):

    """
    Picks randomly amonst options
    """

    def solve(self, gamestate):

        print(f"Solve called -  {self.player}")
        if isinstance(gamestate, GameStatePreGamePlacingPiece):
            piece_type = gamestate.piece_type
            piece = Piece(piece_type, self.player)

            # Randomly choose road from legal placements
            if piece_type == PieceType.road:
                print("Randomly choosing road placement")
                edges = hexgrid.legal_edge_coords()
                prev_settlement = self.game.state.prev_settlement

                valid_edge_placements = list()
                for edge in edges:
                    if prev_settlement in hexgrid.nodes_touching_edge(
                        edge
                    ) and gamestate.game.board.can_place_piece(piece, edge):
                        valid_edge_placements.append(edge)

                gamestate.place_road(random.choice(valid_edge_placements))

            elif piece_type == PieceType.settlement:
                print("Randomly choosing settlement placement")
                nodes = hexgrid.legal_node_coords()
                valid_node_placements = list()

                for node in nodes:
                    if gamestate.game.board.can_place_piece(piece, node):
                        valid_node_placements.append(node)

                gamestate.place_settlement(random.choice(valid_node_placements))

        return 0
