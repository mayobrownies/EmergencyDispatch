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
    def __init__(self, width: int = 20, height: int = 15):
        self.graph = nx.Graph()
        self.nodes = []
        self.width = width
        self.height = height
        self.buildings = set()
        self.roads = set()
        self._generate_city_layout()

    def _generate_city_layout(self):
        self._create_all_nodes()
        self._create_road_network()
        self._add_buildings()
        self._connect_road_intersections()

    def _create_all_nodes(self):
        node_id = 0
        for i in range(self.height):
            for j in range(self.width):
                node = Node(node_id, j, i)
                self.nodes.append(node)
                self.graph.add_node(node_id, pos=(j, i), node_obj=node)
                node_id += 1

    def _create_road_network(self):
        for i in range(0, self.height, 3):
            for j in range(self.width):
                self.roads.add((j, i))

        for j in range(0, self.width, 4):
            for i in range(self.height):
                self.roads.add((j, i))

    def _add_buildings(self):
        for i in range(self.height):
            for j in range(self.width):
                if (j, i) not in self.roads:
                    if random.random() < 0.7:
                        self.buildings.add((j, i))

    def _connect_road_intersections(self):
        for i in range(self.height):
            for j in range(self.width):
                if (j, i) in self.roads:
                    current_id = i * self.width + j

                    if j < self.width - 1 and (j + 1, i) in self.roads:
                        right_id = i * self.width + (j + 1)
                        weight = 1.0
                        self.graph.add_edge(current_id, right_id, weight=weight)

                    if i < self.height - 1 and (j, i + 1) in self.roads:
                        down_id = (i + 1) * self.width + j
                        weight = 1.0
                        self.graph.add_edge(current_id, down_id, weight=weight)

    def get_shortest_path(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            distance = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return [], float('inf')

    def get_random_node(self) -> Node:
        road_nodes = [node for node in self.nodes if (node.x, node.y) in self.roads]
        if road_nodes:
            return random.choice(road_nodes)
        return random.choice(self.nodes)

    def get_nearest_road_to_building(self, building_location: Node) -> Optional[Node]:
        min_distance = float('inf')
        nearest_road = None

        for node in self.nodes:
            if self.is_road(node.x, node.y):
                distance = building_location.get_distance_to(node)
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = node

        return nearest_road

    def get_nearest_station(self, location: Node, stations: List) -> Optional[object]:
        if not stations:
            return None

        min_distance = float('inf')
        nearest_station = None

        for station in stations:
            nearest_road = self.get_nearest_road_to_building(station.location)
            if nearest_road:
                distance = location.get_distance_to(nearest_road)
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
            nearest_road = self.get_nearest_road_to_building(hospital.location)
            if nearest_road:
                distance = location.get_distance_to(nearest_road)
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

    def is_road(self, x: int, y: int) -> bool:
        return (x, y) in self.roads

    def is_building(self, x: int, y: int) -> bool:
        return (x, y) in self.buildings

    def get_road_nodes(self) -> List[Node]:
        return [node for node in self.nodes if (node.x, node.y) in self.roads]