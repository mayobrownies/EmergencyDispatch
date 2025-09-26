import networkx as nx
import random
import numpy as np
from typing import List, Tuple, Optional


class Node:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

    def get_coords(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_distance_to(self, other_node: 'Node') -> float:
        return np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)


class CityGraph:
    def __init__(self, width: int = 10, height: int = 10):
        self.graph = nx.Graph()
        self.nodes = []
        self.width = width
        self.height = height
        self._generate_grid_graph()

    def _generate_grid_graph(self):
        node_id = 0
        for i in range(self.height):
            for j in range(self.width):
                node = Node(node_id, j, i)
                self.nodes.append(node)
                self.graph.add_node(node_id, pos=(j, i), node_obj=node)
                node_id += 1

        for i in range(self.height):
            for j in range(self.width):
                current_id = i * self.width + j

                if j < self.width - 1:
                    right_id = i * self.width + (j + 1)
                    self.graph.add_edge(current_id, right_id, weight=1.0)

                if i < self.height - 1:
                    down_id = (i + 1) * self.width + j
                    self.graph.add_edge(current_id, down_id, weight=1.0)

    def get_shortest_path(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            distance = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return [], float('inf')

    def get_random_node(self) -> Node:
        return random.choice(self.nodes)

    def get_nearest_station(self, location: Node, stations: List) -> Optional[object]:
        if not stations:
            return None

        min_distance = float('inf')
        nearest_station = None

        for station in stations:
            distance = location.get_distance_to(station.location)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station

        return nearest_station

    def get_nearest_hospital(self, location: Node, hospitals: List) -> Optional[object]:
        if not hospitals:
            return None

        min_distance = float('inf')
        nearest_hospital = None

        for hospital in hospitals:
            distance = location.get_distance_to(hospital.location)
            if distance < min_distance:
                min_distance = distance
                nearest_hospital = hospital

        return nearest_hospital

    def get_travel_time(self, path: List[int]) -> float:
        if len(path) < 2:
            return 0.0

        total_time = 0.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i + 1]):
                total_time += self.graph[path[i]][path[i + 1]]['weight']
            else:
                return float('inf')

        return total_time

    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None