"""
Much Like how AI Had a "problem" class, I think it makes sense for us to have one too

TODO: WHAT'S DIFF BETWEEN THIS AND GAME OBJ, like when they actually make a move?
"""

import hexgrid 
import catan.board
from catan_graph import CatanGraph

# Problem we are trying to solve
class Problem:
    # TODO: Take gamestate
    def __init__(self, player=1, n_players=3, graph = None, node_heuristic = None):
        """
        player: Your turn position
        n_players: number of players in the game
        graph: a graph representation of the catan board (see catan_graph.py)
        heuristic: a function that evaluates a node (i.e result of putting a settlement at that node)
        """
        
        self.player = player
        self.nlayers = n_players

        # TODO: Make better default function
        self.node_heuristic = node_heuristic if node_heuristic else (lambda x: 0)

        # Build Catan graph if not specified
        self.graph = graph if graph else CatanGraph(
                                            catan.board.Board(),
                                            [(node_coords, -1) for node_coords in hexgrid.legal_node_coords()],
                                            [(edge_coords, -1) for edge_coords in hexgrid.legal_edge_coords()]
                                         )
        self.legal_settlement_placements = self.graph.get_legal_settlement_placements()

    def get_successors(self):
        pass


test = Problem()
print(len(test.legal_settlement_placements))
    



