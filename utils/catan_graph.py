"""
File to represent catan board as a graph, with nodes as settlements and edges as road positions.

Will need to map nodes to neighbors

Idea is to have classes for nodes and edges, each has values like "is_empty" and "owner", therefore we initialize 
board to "empty" nodes to allow easy traversal and contsraint checking (i.e if we edge / node is already occupied by settlment/road)

TODO:
- Add repr, str values to node and edge based on hexgrid thing to get more friendly debug
- Decide what's exposed to problem, coordinates or nodes / edges themselves? Basically I like idea of exposing the class, but enforce going through graph

"""

# For coordinates
import hexgrid 
import catan.board

# Represents the tiles on the board
class Tile:
    # Resource is one of sheep, wheat, wood, brick, stone, desert
    # Value is the dice roll for that square
    def __init__(self, resource, value):
        self.resource = resource
        self.value = value

# Represents settlement positions
class Node:
    # Tiles contains the tiles that border the node
    # Edges contains the edge coords out from the node
    # Owner contains the player who has built on the node
    def __init__(self, ident, tiles = None, edges = None, owner = -1):
        self.ident = ident
        self.tiles = tiles if tiles else set()
        self.edges = edges if edges else set()
        self.owner = -1
    
    def __str__(self):
        # 1 corresponds to NODE
        return hexgrid.location(1, self.ident)
        
    def __repr__(self):
        # 1 corresponds to NODE
        return hexgrid.location(1, self.ident)
       
    def __hash__(self):
        return hash(self.ident)
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.ident == other.ident
        )

    def can_settle(self):
        return self.owner == -1

    def claim(self, player):
        self.owner = player

    def add_tile(self, tile):
        self.tiles.add(tile)

    def add_edge(self, edge):
        self.edges.add(edge)

# Represents road positions
class Edge:
    # Nodes contains the node coord that this edge connects
    # Player contains the player who has built a road on  this edge
    def __init__(self, ident, nodes = None, owner = -1):
        self.ident = ident
        self.nodes = nodes if nodes else set()
        self.owner = -1
    
    def __str__(self):
        # 0 corresponds to EDGE
        return hexgrid.location(0, self.ident)

    def __repr__(self):
        # 0 corresponds to EDGE
        return hexgrid.location(0, self.ident)

    def __hash__(self):
        return hash(self.ident)
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.ident == other.ident
        )

    def can_build_road(self):
        return self.owner == -1

    def claim(self, player):
        self.owner = player

    def add_node(self, node):
        self.nodes.add(node)

class CatanGraph:
    # Initialize to random graph by default
    def __init__(self, board, nodes, edges):
        # TODO: MAKE THIS COPY by default
        """
        board: catan board, either default catan.board.Board() or TODO: parsed in board
        nodes: list of tuples of form (node_coords, owner)
        edges: list of tuples of form (edge_coords, owner)
        """
        self.board = board

        # Initialize Nodes
        self.nodes = dict()
        for node_coords, owner in nodes:
            self.nodes[node_coords] = Node(node_coords, owner = owner)

        # Initialize edges and connect nodes to edges
        self.edges = dict()
        for edge_coords, owner in edges:
            edge = Edge(edge_coords, owner = owner)

            for node_coords in hexgrid.nodes_touching_edge(edge_coords):
                node = self.nodes[node_coords]
                edge.add_node(node)
                node.add_edge(edge)

            self.edges[edge_coords] = edge
 
        # Add tile data to nodes
        for tile in board.tiles:
            # Get dice roll value of the tile, desert is None
            tile_value = tile.number.value if tile.number.value else -1
            for node_coord in hexgrid.nodes_touching_tile(tile.tile_id):
                self.nodes[node_coord].add_tile(Tile(tile.terrain, tile_value))

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
            return node.edges
        elif isinstance(node, int):
            return self.nodes[node].edges
        else:
            raise Exception("Unknown type passed into get_edges_of_node()")
    
    def get_nodes_of_edge(self, edge):
        """
        Returns nodes of an Edge
        edge: Edge or Edge Coordinates
        """
        if isinstance(edge, Edge):
            return edge.nodes
        elif isinstance(edge, int):
            return self.edges[edge].nodes
        else:
            raise Exception("Unknown type passed into get_nodes_of_edge()")
    
    def get_neighbors(self, node, check_ownership = False):
        """
        Returns valid (matching player) neighbor nodes
        node: Node or Node Coordinates
        check_ownership: Whether to filter neighbors on ownership (i.e do not include nodes you don't own, or can't get to)
        """
        node = self.get_node(node) # Could be coords, could be node
        if check_ownership:
            edges = filter(
                    lambda e: e.owner == node.owner, 
                    self.get_edges_of_node(node)
                ) 
        else:
            edges = self.get_edges_of_node(node) 

        neighbors = []
        
        for edge in edges:
            for neighbor in self.get_nodes_of_edge(edge):
                # Remove original node
                if neighbor != node:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def get_legal_settlement_placements(self):
        """
        Return legal nodes where players can place settlements
        """
        legal_node_placements = set()
        illegal_node_placements = set()

        for _, v in self.nodes.items():
            # Add unsettled nodes to legal nodes
            if v.can_settle():
                legal_node_placements.add(v)
            # Add settled node's neighbors to illegal
            else:
                for neighbor in self.get_neighbors(v):
                    illegal_node_placements.add(neighbor)

        # Legal nodes is then diff between sets
        return legal_node_placements.difference(illegal_node_placements)

