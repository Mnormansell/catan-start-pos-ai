# import hexgrid

# """
# Class to simplify graph traversal (i.e checking node neighbors)
# """
# class CatanGraph():
#     def __init__(self):
#         # Initialize edges and nodes
#         self.nodes = {node_coords: set() for node_coords in hexgrid.legal_node_coords()}
#         self.edges = {edge_coords: set() for edge_coords in hexgrid.legal_edge_coords()}

#         # Connect graph
#         for edge_coords in self.edges.keys():
#             for node_coords in hexgrid.nodes_touching_edge(edge_coords):
#                 self.edges[edge_coords].add(node_coords)
#                 self.nodes[node_coords].add(edge_coords)

#     def get_edges_of_node(self, node_coords):
#         return self.nodes[node_coords]

#     def get_nodes_of_edge(self, edge_coords):
#         return self.edges[edge_coords]

#     def get_node_neighbors(self, node_coords):
#         neighbors = set()
#         for edge_coords in self.get_edges_of_node(node_coords):
#             for node in self.get_nodes_of_edge(edge_coords):
#                 neighbors.add(node)
#         neighbors.remove(node_coords)
#         return neighbors

# For coordinates
import hexgrid
import catan.board


class Node:
    """
    A node is position where a settlement can be placed, contains info about outgoing edges (roads)
    and an owner. 
    """

    def __init__(self, coords, tiles=None, edges=None, owner=-1):
        self.coords = coords
        self.tiles = tiles if tiles else set()
        self.edges = edges if edges else set()
        self.owner = owner

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

    def occupied(self):
        return self.owner == -1

    def claim(self, player):
        self.owner = player

    def add_tile(self, tile):
        self.tiles.add(tile)

    def add_edge(self, edge):
        self.edges.add(edge)


class Edge:
    """
    An edge is a position where a road can be placed, contains info about connected nodes (settlements)
    and an owner. 
    """

    def __init__(self, coords, nodes=None, owner=-1):
        self.coords = coords
        self.nodes = nodes if nodes else set()
        self.owner = -1

    def __str__(self):
        # 0 corresponds to EDGE
        return hexgrid.location(0, self.coords)

    def __repr__(self):
        # 0 corresponds to EDGE
        return hexgrid.location(0, self.coords)

    def __hash__(self):
        return hash(self.coords)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.coords == other.coords
        )

    def occupied(self):
        return self.owner == -1

    def claim(self, player):
        self.owner = player

    def add_node(self, node):
        self.nodes.add(node)


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
        nodes: list of tuples of form (node_coords, owner)
        edges: list of tuples of form (edge_coords, owner)
        """
        self.board = board

        self.nodes = {node_coords: Node(node_coords)
                      for node_coords in hexgrid.legal_node_coords()}
        self.edges = {edge_coords: Edge(edge_coords)
                      for edge_coords in hexgrid.legal_edge_coords()}

        # Connect graph
        for edge_coords, edge in self.edges.items():
            for node_coords in hexgrid.nodes_touching_edge(edge_coords):
                node = self.nodes[node_coords]
                edge.add_node(node)
                node.add_edge(edge)

            self.edges[edge_coords] = edge

        # Add tiles to node
        for tile in board.tiles:
            # Get dice roll value of the tile, desert is None
            for node_coord in hexgrid.nodes_touching_tile(tile.tile_id):
                self.nodes[node_coord].add_tile(tile)

        # Set pieces appropriately
        self.set_pieces(self.board.pieces)

    def set_pieces(self, pieces_dict):
        for (piece_type, coord), piece in pieces_dict.items():
            # If piece is an edge
            if piece_type == 0:
                self.edges[coord].owner = piece.owner
            # If piece is a node
            elif piece_type == 1:
                self.nodes[coord].owner = piece.owner
            # Ignore robber
            else:
                print(piece)

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

    def get_edge(self, edge):
        """
        Returns Edge given Edge or Edge Coordinates
        """
        """
        Returns nodes of an Edge
        edge: Edge or Edge Coordinates
        """
        if isinstance(edge, Edge):
            return edge
        elif isinstance(edge, int):
            return self.edges[edge]
        else:
            raise Exception("Unknown type passed get_edge get_edge_nodes()")

    def get_edges_of_node(self, node):
        """
        Returns edges of a Node
        node: Node or Node Coordinates
        """
        if isinstance(node, Node):
            return [edge.coords for edge in node.edges]
        elif isinstance(node, int):
            return [edge.coords for edge in self.nodes[node].edges]
        else:
            raise Exception("Unknown type passed into get_edges_of_node()")

    def get_nodes_of_edge(self, edge):
        """
        Returns nodes of an Edge
        edge: Edge or Edge Coordinates
        """
        if isinstance(edge, Edge):
            return [node.coords for node in edge.nodes]
        elif isinstance(edge, int):
            return [node.coords for node in self.edges[edge].nodes]
        else:
            raise Exception("Unknown type passed into get_nodes_of_edge()")

    def get_node_neighbors(self, node, check_ownership=False):
        """
        Returns coords of neighbor nodes (filtered or unfiltered)
        node: Node or Node Coordinates
        check_ownership: Whether to filter neighbors on ownership (i.e do not include nodes other players own / cannot get to)
        """
        node = self.get_node(node)  # Could be coords, could be node
        edges = self.get_edges_of_node(node)

        neighbors = set()
        for edge_coords in self.get_edges_of_node(node):
            # If we want to filter on ownership, then only consider edges where it's empty or shares ownership with node
            edge = self.edges[edge_coords]
            if not check_ownership or edge.owner == -1 or edge.owner == node.owner:
                for node_coords in self.get_nodes_of_edge(edge_coords):
                    # Need to ignore current node, and if filtering, again only consider empty node's or ones with shared ownership
                    neighbor_node = self.nodes[node_coords]
                    if neighbor_node.coords != node.coords and \
                            (not check_ownership or neighbor_node.owner == -1 or neighbor_node.owner == node.owner):
                        neighbors.add(node_coords)

        return neighbors

        # def get_node_neighbors(self, node_coords):
#         neighbors = set()
#         for edge_coords in self.get_edges_of_node(node_coords):
#             for node in self.get_nodes_of_edge(edge_coords):
#                 neighbors.add(node)
#         neighbors.remove(node_coords)
#         return neighbors

    def get_legal_settlement_placements(self):
        """
        Return legal nodes where players can place settlements
        """
        legal_node_placements = set()
        illegal_node_placements = set()

        for _, v in self.nodes.items():
            # Add unsettled nodes to legal nodes
            if v.occupied():
                legal_node_placements.add(v)
            # Add settled node's neighbors to illegal
            else:
                for neighbor in self.get_neighbors(v):
                    illegal_node_placements.add(neighbor)

        # Legal nodes is then diff between sets
        return legal_node_placements.difference(illegal_node_placements)
