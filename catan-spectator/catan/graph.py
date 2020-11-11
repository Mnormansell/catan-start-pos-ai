import hexgrid
import catan.board
from catan.pieces import Piece, PieceType

class Node:
    """
    A node is position where a settlement can be placed, contains info about outgoing edges (roads)
    and an owner.
    Using a class here as we need to store tiles 
    """

    def __init__(self, coords, tiles=None, edges=None):
        self.coords = coords
        self.tiles = tiles if tiles else set()
        self.edges = edges if edges else set()

    def __str__(self):
        # 1 corresponds to NODE
        return hexgrid.location(1, self.coords)

    def __repr__(self):
        # 1 corresponds to NODE
        return hexgrid.location(1, self.coords)

    def __hash__(self):
        return hash(self.coords)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coords == other.coords
        )
    
    def add_tile(self, tile):
        self.tiles.add(tile)

    def add_edge(self, edge):
        self.edges.add(edge)

class CatanGraph:
    """
    This class contains some duplicate information as the board but with less overhead. The idea
    is to gain functionality (like get edges of a node, get neighbords easily) while allowing an
    AI to modify this "lightweight" class instead of modifying the entire board.
    """

    def __init__(self, board):
        # TODO: MAKE THIS COPY by default
        """
        board: catan board, either default catan.board.Board() or TODO: parsed in board
        """
        self.board = board

        self.nodes = {node_coords: Node(node_coords)
                      for node_coords in hexgrid.legal_node_coords()}
        # For edges we only need to store the bordering Nodes, no need for a class
        self.edges = {edge_coords: set()
                      for edge_coords in hexgrid.legal_edge_coords()}

        # Connect graph
        for edge_coords in self.edges.keys():
            for node_coords in hexgrid.nodes_touching_edge(edge_coords):
                node = self.nodes[node_coords]
                self.edges[edge_coords].add(node)
                node.add_edge(edge_coords)


        # Add tiles to node
        for tile in board.tiles:
            # Get dice roll value of the tile, desert is None
            for node_coord in hexgrid.nodes_touching_tile(tile.tile_id):
                self.nodes[node_coord].add_tile(tile)

    def get_node(self, node):
        """
        Returns Node given Node or Node Coordinates
        """
        if isinstance(node, Node):
            return node
        elif isinstance(node, int):
            return self.nodes[node]
        else:
            raise Exception("Unknown type passed into get_node()")

    def get_edges_of_node(self, node):
        """
        Returns edges of a Node
        node: Node or Node Coordinates
        """
        if isinstance(node, Node):
            return node.edges
        elif isinstance(node, int):
            return self.nodes[node].edges
        else:
            raise Exception("Unknown type passed into get_edges_of_node()")

    def get_nodes_of_edge(self, edge):
        """
        Returns nodes of an Edge
        edge:  Edge Coordinates
        """
        return [node.coords for node in self.edges[edge]]

    def get_node_neighbors(self, node, pieces, check_ownership=None):
        """
        Returns coords of neighbor nodes (filtered or unfiltered)
        node: Node or Node Coordinates
        pieces: Pieces dictionary
        check_ownership: Player to check ownership for
        """
        node = self.get_node(node)  # Could be coords, could be node
        edges = self.get_edges_of_node(node)

        neighbors = set()
        for edge_coords in self.get_edges_of_node(node):
            # 3 checks on the edges of the node we are looking at
            # 1: If we aren't looking to check ownership we loop over that edge's nodes
            # 2: If the edge is not currently owner (not in the pieces dict)
            # 3: If the edge is owner by the player we are cheking ownership for 
            if check_ownership is None or pieces.get((0, edge_coords)) is None or pieces[(0, edge_coords)].owner == check_ownership:
                for node_coords in self.get_nodes_of_edge(edge_coords):                 
                    # Same 3 checks for the node coords as we did with edge coords
                    if check_ownership is None or pieces.get((1, node_coords)) is None or pieces[(1, node_coords)].owner == check_ownership:
                        neighbors.add(node_coords)
        # Remove current node from set
        neighbors.remove(node.coords)
        return neighbors


    def get_legal_settlement_placements(self, pieces):
        """
        Return legal nodes where players can place settlements
        """
        # Start with all possible placements
        legal_node_placements = hexgrid.legal_node_coords()

        # Look to remove placements based on settlements currently placed in pieces dict
        for (piece_type, coords) in pieces.keys():
            # If a node, remove the cords from the legal coords
            if piece_type == 1:
                # Remove that settlement coords and it's neighbors
                legal_node_placements.remove(coords)
                for node_coords in self.get_node_neighbors(coords, pieces):
                     legal_node_placements.discard(node_coords)

        # Legal nodes is then diff between sets
        return legal_node_placements
