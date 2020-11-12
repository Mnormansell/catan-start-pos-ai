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

    def __init__(self, players, graph):
        self.players = players
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

        # Create new copies of pieces and legal placements with that node claimed for the player
        def copy_state_without_node(pieces, legal_placements, node, player):
            pieces_ = copy.copy(pieces)
            legal_placements_ = copy.copy(legal_placements)

            pieces_[(1, node)] = Piece(PieceType.settlement, player)
            legal_placements_.remove(node)
            for neighbor in graph.get_node_neighbors(node, pieces_):
                legal_placements_.discard(neighbor)
            
            return pieces_, legal_placements_

        # TODO: Implement pruning, limiting number of nodes?
        def tree_search(cur_turn, end_turn, pieces, legal_placements, alphas):
            tree_search.counter[cur_turn] += 1
            # If last turn we evaluate at our second level
            if cur_turn == end_turn:
                best_placement_hval, best_placement_coords = max(
                    map(
                        lambda node: (heuristic.evaluate_node(node, pieces, cur_turn), node),
                        legal_placements
                    )
                )
                ret = [(-1, None) for i in range(len(players))]
                ret[players.index(snake[cur_turn])] = (best_placement_hval, best_placement_coords)
                return ret
            
            players_index = players.index(snake[cur_turn])
            
            # First turn - Evaluate each node appropriately
            if cur_turn < len(players):
                best_settlement_hval = -1
                best_settlement_coords = -1
                ret = [(-1, None) for i in range(len(players))] # Default initialize to lameo

                # print(legal_placements)
                for node in legal_placements:
                    first_placement_hval = heuristic.evaluate_node(node, pieces, cur_turn) 
                    
                    # Create new copies of pieces and legal placements, and claim that node and recurse     
                    pieces_, legal_placements_ = copy_state_without_node(pieces, legal_placements, node, players[players_index])

                    # We prune if best second settlement placement yields a worse combination than one we've already calculated 
                    # This follows with the assumption that heuristics for settlements can only get worse with other player choices
                    second_placement_hval = max(
                        map(
                            lambda n: heuristic.evaluate_node(n, pieces, cur_turn),
                            legal_placements
                        )
                    )
                    if first_placement_hval + second_placement_hval <= alphas[players_index]:
                        continue

                    # Calculate lower leaves of the tree
                    if cur_turn == start_turn:
                        alphas_ = [-float("inf") for i in range(len(players))]
                        alphas_[players_index] = alphas[players_index]
                        ret_ = tree_search(cur_turn+1, end_turn, pieces_, legal_placements_, alphas_)
                    else:
                        ret_ = tree_search(cur_turn+1, end_turn, pieces_, legal_placements_, alphas)
                    second_level_hval, _ = ret_[players_index] # Get second level hval

                    # Update alpha (if it wasn't pruned)
                    hval = first_placement_hval + second_level_hval
                    
                    if hval > alphas[players_index]:
                        alphas[players_index] = hval 
                    # else:
                    #     print("Not greater")
                    
                    # Find max correctly
                    if hval > best_settlement_hval:
                        best_settlement_hval = hval
                        best_settlement_coords = node
                        ret = ret_
                
                return ret
            # Second turn - Pick best option
            else:
                best_placement_hval, best_placement_coords = max(
                    map(
                        lambda node: (heuristic.evaluate_node(node, pieces, cur_turn), node),
                        legal_placements
                    )
                )    
                
                pieces_, legal_placements_ = copy_state_without_node(pieces, legal_placements, best_placement_coords, players[players_index])

                ret = tree_search(cur_turn+1, end_turn, pieces_, legal_placements_, alphas)
                ret[players_index] = (best_placement_hval, best_placement_coords)
                return ret
        

        tree_search.counter = {i: 0 for i in range(len(snake))}
        # Initialize tree search, alphas to negative infinity
        ret_arr = tree_search(start_turn, end_turn, pieces, legal_placements, [-float("inf") for i in range(len(players))])
        _, best_placement_coords = ret_arr[players.index(snake[start_turn])]
        # print(tree_search.counter)
        
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
                    GeneralHeuristic(gamestate.game.players, gamestate.game.board.graph)
                )
                gamestate.place_settlement(best_settlement)

        return 0
