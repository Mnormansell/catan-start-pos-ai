"""
File that stores collection of agents, i.e random agent, 
"""
from catan.states import GameStatePreGamePlacingPiece
from catan.pieces import Piece, PieceType
from catan.graph import CatanGraph
from catan.board import Terrain
import hexgrid
import random
import copy
from queue import Queue

class Heuristic(object):
    """
    An abstract class for node heuristics to inherit from
    """

    def __init__(self, players, graph):
        self.players = players
        self.graph = graph
    
    def number_to_prob(self, number):
        """
        Convert number on tile (2..12) to probability of being rolled
        """
        if number is None:
            return 0
        
        if number <= 7: 
            return (number-1)/36
        
        return ((-number % 7) - 1) / 36

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
    def evaluate_node(self, node, turn, pieces):
        if turn < 2 * len(self.players):
            player = (self.players + list(reversed(self.players)))[turn]
        else:
            player = self.players[turn % len(self.players)]

        resources = {terrain : 0 for terrain in Terrain}
        tiles = []
        val = 0

        # Calculate what resources player already owns
        for (piece_type, coord), piece in pieces.items():
            if piece_type == 1 and piece.owner == player:
                for tile in self.graph.nodes[coord].tiles:
                    tiles.append(tile)
                    resources[tile.terrain] += 1
        for tile in self.graph.nodes[node].tiles:
            tiles.append(tile)
            resources[tile.terrain] += 1

        # Formula IDEA: Weight probability of rolling by number of that resources already owned (larger weight for less resources owned)
        for tile in tiles:
            val += (1 / (1 + resources[tile.terrain])) * (self.number_to_prob(tile.number.value) * 36)


        return val

class TestingHeuristic(Heuristic):
    # General Evaluation -> just try to get every card with favorable rolls
    def evaluate_node(self, node, turn, pieces):
        if turn < 2 * len(self.players):
            player = (self.players + list(reversed(self.players)))[turn]
        else:
            player = self.players[turn % len(self.players)]

        weights = {
            Terrain.wood: 3,            
            Terrain.brick: 3,
            Terrain.wheat: 0,
            Terrain.sheep: 0,
            Terrain.ore: 0,           
            Terrain.desert: 0
        }
        resources = {terrain : 0 for terrain in Terrain}
        tiles = []
        val = 0

        # Calculate what resources player already owns
        for (piece_type, coord), piece in pieces.items():
            if piece_type == 1 and piece.owner == player:
                for tile in self.graph.nodes[coord].tiles:
                    tiles.append(tile)
                    resources[tile.terrain] += 1
        for tile in self.graph.nodes[node].tiles:
            tiles.append(tile)
            resources[tile.terrain] += 1

        # Formula IDEA: Weight probability of rolling by number of that resources already owned (larger weight for less resources owned)
        for tile in tiles:
            val += (weights[tile.terrain] / (1 + resources[tile.terrain])) * (self.number_to_prob(tile.number.value) * 36)


        return val

class Agent(object):
    """
    An abstract class for other agents to inherit from

    member variables:
    game -> Catan game object
    player -> Player that the agent belongs to
    heuristic -> Heuristic (Eval function) for the agent to use
    tile_cutoff -> Only consider nodes with > this many tiles. Default 1 (no cutoff)
    """

    def __init__(self, game, player, heuristic = None, tile_cutoff = 0):
        self.game = game
        self.player = player
        self.heuristic = heuristic
        self.tile_cutoff = tile_cutoff

    # Solve at that state
    def solve(self, gamestate):
        raise NotImplemented()

    # If first turn, recurse till second turn to find best 1st / 2nd settlement combos
    def calc_best_settlement_placement(self, gamestate):
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
            players_index = players.index(snake[cur_turn])
            
            # If last turn we evaluate at our second level
            if cur_turn == end_turn:
                best_placement_hval, best_placement_coords = max(
                    map(
                        lambda node: (self.heuristic.evaluate_node(node, cur_turn, pieces), node),
                        legal_placements
                    )
                )
                ret = [(-1, None) for i in range(len(players))]
                ret[players_index] = (best_placement_hval, best_placement_coords)
                return ret
            
            
            # First turn - Evaluate each node appropriately
            if cur_turn < len(players):
                best_settlement_hval = -1
                best_settlement_coords = -1
                ret = [(-1, None) for i in range(len(players))] # Default initialize to 

                for node in legal_placements:
                    first_placement_hval = self.heuristic.evaluate_node(node, cur_turn, pieces) 
                    
                    # Create new copies of pieces and legal placements, and claim that node and recurse     
                    pieces_, legal_placements_ = copy_state_without_node(pieces, legal_placements, node, players[players_index])

                    # We prune if best second settlement placement yields a worse combination than one we've already calculated 
                    # This follows with the assumption that heuristics for settlements can only get worse with other player choices
                    second_placement_hval = max(
                        map(
                            lambda n: self.heuristic.evaluate_node(n, cur_turn, pieces_),
                            legal_placements_
                        )
                    )
                    if first_placement_hval + second_placement_hval <= alphas[players_index]:
                        # print(alphas[players_index], first_placement_hval, second_placement_hval)
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
                        lambda node: (self.heuristic.evaluate_node(node, cur_turn, pieces), node),
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
        print(tree_search.counter)
        return best_placement_coords

    # BFS to closest desired node / port
    def calc_best_road_placement(self, gamestate):

        if gamestate.is_in_pregame():
            turn = gamestate.game._cur_turn

            graph = gamestate.game.board.graph
            pieces = gamestate.game.board.pieces
            sorted_legal_placements = list(
                sorted(
                    graph.get_legal_settlement_placements(pieces),
                    key = lambda x: self.heuristic.evaluate_node(x, turn, pieces),
                    reverse = True
                )
            )

            path = bfs(
                gamestate.prev_settlement, 
                gamestate.game.board, 
                sorted_legal_placements[:2], 
                gamestate.game.get_cur_player()
            )

            if len(path) == 0:
                raise Exception("No path to goals found!")
            return path[1][1]

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
    # Initialize to slightly higher cutoff, default to GeneralHeuristic
    def __init__(self, game, player, heuristic = "GeneralHeuristic", tile_cutoff = 1):
        super(NaiveAgent, self).__init__(game, player, heuristic, tile_cutoff)

    def calc_best_settlement_placement(self, gamestate):
        graph = gamestate.game.board.graph
        pieces = gamestate.game.board.pieces
        return max( 
            map(
                lambda node: (
                    self.heuristic.evaluate_node(node, gamestate.game._cur_turn, pieces), 
                    node
                ),
                filter(
                    lambda node: len(graph.nodes[node].tiles) > self.tile_cutoff, 
                    graph.get_legal_settlement_placements(pieces)
                )
            )
        )[1]

    def solve(self, gamestate):
        if isinstance(gamestate, GameStatePreGamePlacingPiece):
            piece_type = gamestate.piece_type
            piece = Piece(piece_type, self.player)

            # Choose best available option
            if piece_type == PieceType.road:
                best_road = self.calc_best_road_placement(gamestate)
                # Road will be second item of second item in path
                gamestate.place_road(best_road)
            
            elif piece_type == PieceType.settlement:
                # Brute force to find best settlement
                best_settlement = self.calc_best_settlement_placement(gamestate)
                gamestate.place_settlement(best_settlement)

        return 0

class TreeAgent(Agent):
    # Initialize to slightly higher cutoff, default to GeneralHeuristic
    def __init__(self, game, player, heuristic = "GeneralHeuristic", tile_cutoff = 1):
        super(TreeAgent, self).__init__(game, player, heuristic, tile_cutoff)

    def solve(self, gamestate):
        if isinstance(gamestate, GameStatePreGamePlacingPiece):
            piece_type = gamestate.piece_type
            piece = Piece(piece_type, self.player)

            # Randomly choose road from legal placements
            if piece_type == PieceType.road:
                best_road = self.calc_best_road_placement(gamestate)
                # Road will be second item of second item in path
                gamestate.place_road(best_road)

            elif piece_type == PieceType.settlement:
                # Brute force to find best settlement
                best_settlement = self.calc_best_settlement_placement(gamestate)
                gamestate.place_settlement(best_settlement)

        return 0


def bfs(node, board, goals, player=None):
    """
    Traverse from node until we reach a goal state "with backtracking"

    @param node: starting node coordinates
    @param board: Catan board
    @param goals: list of goal nodes, stop when first is hit
    @param player: whether to filter results based on legal moves for player
    returns path (of nodes and edges to goal)
    """
    pieces = board.pieces
    graph = board.graph

    seen = set()
    backtrack = dict()
    q = Queue() # Queue stores nodes

    # Initialize with starting node
    q.put(node)
    seen.add((1, node))
    backtrack[(1, node)] = None

    while not q.empty():
        cur_node = q.get()

        # If a goal, backtrack
        if cur_node in goals:
            # Return list
            path = []
            # Backtrack to current states "parent"

            # In form (piece_type, coords)
            parent = backtrack[(1, cur_node)]

            while parent:
                # Append parent->cur action
                path.append(parent)
                # Backtrack from parent state
                parent = backtrack[parent]
            # Need to reverse list to put actions in correct order
            return path[::-1]

        for edge_coords in graph.get_edges_of_node(cur_node):
            # 3 checks on the edges of the node we are looking at
            # 1: If we aren't looking to check ownership we loop over that edge's nodes
            # 2: If the edge is not currently owner (not in the pieces dict)
            # 3: If the edge is owner by the player we are cheking ownership for 
            if (0, edge_coords) in seen:
                continue

            if player is None or pieces.get((0, edge_coords)) is None or pieces[(0, edge_coords)].owner == player:
                seen.add((0, edge_coords))
                backtrack[(0, edge_coords)] = (1, cur_node)

                # Get successor node (aka the one out of the two nodes connected to the edge that isn't our initial node)
                suc_node = list(filter(
                    lambda x: x != cur_node,
                    graph.get_nodes_of_edge(edge_coords)
                ))[0]

                if suc_node in (1, seen):
                    continue

                # Same cchecks for nodes
                if player is None or pieces.get((1, suc_node)) is None or pieces[(1, suc_node)].owner == player:
                    seen.add((1, suc_node))
                    backtrack[(1, suc_node)] = (0, edge_coords)
                    q.put(suc_node)

    return []

# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     # Initialize overhead
#     seen = set()
#     backtrack = dict() # Dict to build actions to goal, form "Child: (Parent, action: Parent->Child)" 
#     fringe = util.Queue()
    
#     # Prepare for BFS on start node
#     cur = problem.getStartState()
#     seen.add(cur)
#     fringe.push(cur)

#     # Populate backtrack
#     backtrack[cur] = () # Backtracks to nothing

#     # Iterate until stack is empty
#     while not fringe.isEmpty():
#         # Pop most recently seen element
#         cur = fringe.pop()

#         # If goal state, backtrack to form return state and return
#         if problem.isGoalState(cur):
#             # Return list
#             actions = []
#             # Backtrack to current states "parent"
#             parent = backtrack[cur]

#             while len(parent) != 0:
#                 # Append parent->cur action
#                 actions.append(parent[1])

#                 # Backtrack from parent state
#                 parent = backtrack[parent[0]]
            
#             # Need to reverse list to put actions in correct order
#             return actions[::-1]


#         # If not goal state, get Successors and DFS more
#         successors = problem.getSuccessors(cur)

#         for successor in successors:
#             # Successor has form [State, Action, Weight]
#             state = successor[0]

#             # If not, setup backtracking and append
#             if state not in seen:
#                 # Mark as seen
#                 seen.add(state)

#                 # Add backtracking element, (parent, action parent->child)
#                 backtrack[state] = (cur, successor[1])
                
#                 # Push onto BFS fringe
#                 fringe.push(state)

#     # Hit this if no goal state?
#     util.raiseNotDefined()