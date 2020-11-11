"""
File that stores collection of agents, i.e random agent, 
"""
from catan.states import GameStatePreGamePlacingPiece
from catan.pieces import Piece, PieceType
from catan.graph import CatanGraph
import hexgrid
import random
import copy

class Heuristic(object):
    """
    An abstract class for node heuristics to inherit from
    """

    def __init__(self, player, graph):
        self.player = player
        self.graph = graph
    
    # Evaluate a node for a specific turn
    def evaluate_node(self, node, turn, pieces):
        """
        param node: which node to evaluate
        param turn: 1 or 2 representing first turn or second turn
        param pieces: pieces (state of game essentially) on which to evaluate
        """
        raise NotImplemented()

    # Evaluate an edge placement (Eval same for different turns)
    def evaluate_edge(self, edge, pieces):
        """
        param edge: which edge to evaluate
        param pieces: pieces (state of game essentially) on which to evaluate
        """
        raise NotImplemented()

class GeneralHeuristic(Heuristic):
    # General Evaluation -> just try to get every card with favorable rolls
    def evaluate_node(self, node, pieces, turn):
        val = 0
        for tile in self.graph.nodes[node].tiles:
            die_roll = tile.number.value
            if die_roll == 2 or die_roll == 12:
                val += 1
            elif die_roll == 3 or die_roll == 11:
                val += 2
            elif die_roll == 4 or die_roll == 10:
                val += 3
            elif die_roll == 5 or die_roll == 9:
                val += 4
            elif die_roll == 6 or die_roll == 8:
                val += 5
            elif die_roll == 7:
                val += 0
        return val

class Agent(object):
    """
    An abstract class for other agents to inherit from

    member variables:
    game -> Catan game object
    player -> Player that the agent belongs to
    tile_cutoff -> Only consider nodes with > this many tiles. Default 1 (no cutoff)
    """

    def __init__(self, game, player, tile_cutoff = 0):
        self.game = game
        self.player = player
        self.tile_cutoff = tile_cutoff

    # Solve at that state
    def solve(self, gamestate):
        raise NotImplemented()

    # If first turn, recurse till second turn to find best 1st / 2nd settlement combos
    def calc_best_settlement_placement(self, gamestate, heuristic):
        players = gamestate.game.players
        snake = players + list(reversed(players))
        start_turn = gamestate.game._cur_turn
        end_turn = (len(snake) - 1) - players.index(self.player)

        graph = gamestate.game.board.graph
        pieces = gamestate.game.board.pieces
        legal_placements = set(
            filter(
                lambda node: len(graph.nodes[node].tiles) > self.tile_cutoff, 
                graph.get_legal_settlement_placements(pieces)
            )
        )

        # TODO: Implement pruning, limiting number of nodes?
        def tree_search(cur_turn, end_turn, pieces, legal_placements):
            tree_search.counter[cur_turn] += 1
            # If last turn we evaluate at our second level
            if cur_turn == end_turn:
                return max(
                    map(
                        lambda node: (heuristic.evaluate_node(node, pieces, cur_turn), node),
                        legal_placements
                    )
                )                
            
            # First turn - Evaluate each node appropriately
            if cur_turn < len(players):
                best_settlement_hval = -1
                best_settlement_coords = -1

                for node in legal_placements:
                    # Create new copies of pieces and legal placements, and claim that node and recurse     
                    pieces_ = copy.copy(pieces)
                    legal_placements_ = copy.copy(legal_placements)

                    pieces_[(1, node)] = Piece(PieceType.settlement, players[cur_turn])
                    legal_placements_.remove(node)
                    for neighbor in graph.get_node_neighbors(node, pieces_):
                        legal_placements_.discard(neighbor)

                    hval, _ = tree_search(cur_turn+1, end_turn, pieces_, legal_placements_)

                    # If first turn for our player, we need to calcualte the heuristic of the first node as well
                    # TODO: if we do pruning, we may need ot pass this an an argument
                    if players[cur_turn] == self.player:
                        hval += heuristic.evaluate_node(node, pieces, cur_turn) 

                    # Find max correctly
                    if hval > best_settlement_hval:
                        best_settlement_hval = hval
                        best_settlement_coords = node
                
                return best_settlement_hval, best_settlement_coords
            # Second turn - Pick best option
            else:
                _, best_placement_coords = max(
                    map(
                        lambda node: (heuristic.evaluate_node(node, pieces, cur_turn), node),
                        legal_placements
                    )
                )    
                
                pieces_ = copy.copy(pieces)
                legal_placements_ = copy.copy(legal_placements)

                pieces_[(1, best_placement_coords)] = Piece(PieceType.settlement, snake[cur_turn])
                legal_placements_.remove(best_placement_coords)
                for neighbor in graph.get_node_neighbors(best_placement_coords, pieces_):
                    legal_placements_.discard(neighbor)

                return tree_search(cur_turn+1, end_turn, pieces_, legal_placements_)
        
        tree_search.counter = {i: 0 for i in range(len(snake))}
        _, best_placement_coords = tree_search(start_turn, end_turn, pieces, legal_placements)
        print(tree_search.counter)

        return best_placement_coords

    # BFS to closest desired node / port
    def calc_best_road_placement(self, heuristic):
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

class NaiveAgent(Agent):
    # Initialize to slightly higher cutoff
    def __init__(self, game, player, tile_cutoff = 1):
        super(NaiveAgent, self).__init__(game, player, tile_cutoff)

    def solve(self, gamestate):
        # # Get turn
        # if gamestate.game._cur_turn < len(gamestate.game.players):
        #     print("First Turn")
        # else:
        #     print("Second Turn")

        if isinstance(gamestate, GameStatePreGamePlacingPiece):
            piece_type = gamestate.piece_type
            piece = Piece(piece_type, self.player)

            print(gamestate.game._cur_turn)

            # Randomly choose road from legal placements
            if piece_type == PieceType.road:
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
                # Brute force to find best settlement
                best_settlement = self.calc_best_settlement_placement(
                    gamestate, 
                    GeneralHeuristic(self.player, gamestate.game.board.graph)
                )
                gamestate.place_settlement(best_settlement)

        return 0
