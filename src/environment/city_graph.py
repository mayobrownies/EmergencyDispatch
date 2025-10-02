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
        self.traffic_multipliers = {}
        self.blocked_roads = set()
        self.traffic_events = []
        self._generate_city_layout()
        self._initialize_traffic_system()

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

    def get_traffic_aware_path(self, start_node: int, end_node: int) -> Tuple[List[int], float]:
        try:
            temp_graph = self.graph.copy()

            for road_coord in self.blocked_roads:
                for node_id in temp_graph.nodes():
                    node = self.get_node_by_id(node_id)
                    if node and (node.x, node.y) == road_coord:
                        edges_to_remove = list(temp_graph.edges(node_id))
                        temp_graph.remove_edges_from(edges_to_remove)

            path = nx.shortest_path(temp_graph, start_node, end_node, weight='weight')

            total_time = 0.0
            for i in range(len(path) - 1):
                current_node = self.nodes[path[i]]
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data and 'weight' in edge_data:
                    base_time = edge_data['weight']
                    road_coord = (current_node.x, current_node.y)
                    traffic_multiplier = self.traffic_multipliers.get(road_coord, 1.0)
                    total_time += base_time * traffic_multiplier
                else:
                    return [], float('inf')

            return path, total_time
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

    def _initialize_traffic_system(self):
        for road in self.roads:
            self.traffic_multipliers[road] = 1.0

    def update_traffic_conditions(self, current_time: float):
        self._update_dynamic_traffic(current_time)
        self._process_traffic_events(current_time)

    def _update_dynamic_traffic(self, current_time: float):
        rush_hour_multiplier = self._get_rush_hour_multiplier(current_time)

        for road in self.roads:
            if road not in self.blocked_roads:
                base_traffic = 1.0
                random_variation = random.uniform(0.7, 1.3)
                traffic_level = base_traffic * rush_hour_multiplier * random_variation

                if traffic_level < 1.2:
                    self.traffic_multipliers[road] = random.uniform(0.8, 1.1)
                elif traffic_level > 2.0:
                    self.traffic_multipliers[road] = random.uniform(2.0, 3.0)
                else:
                    self.traffic_multipliers[road] = traffic_level

    def _get_rush_hour_multiplier(self, current_time: float) -> float:
        hour = (current_time / 3600.0) % 24

        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 2.5
        elif 6 <= hour <= 10 or 16 <= hour <= 20:
            return 1.8
        elif 11 <= hour <= 15:
            return 1.2
        else:
            return 0.7

    def add_traffic_incident(self, duration: float, severity: str = "moderate", start_time: float = 0):
        road_nodes = list(self.roads)
        if road_nodes:
            affected_road = random.choice(road_nodes)

            if severity == "severe":
                self.blocked_roads.add(affected_road)
                multiplier = float('inf')
            elif severity == "moderate":
                multiplier = 3.0
            else:
                multiplier = 1.5

            self.traffic_events.append({
                'road': affected_road,
                'start_time': start_time,
                'duration': duration,
                'multiplier': multiplier,
                'severity': severity,
                'created': True
            })

    def _process_traffic_events(self, current_time: float):
        active_events = []

        for event in self.traffic_events:
            event_end = event['start_time'] + event['duration']

            if current_time <= event_end:
                road = event['road']
                if event['severity'] == "severe":
                    self.blocked_roads.add(road)
                else:
                    self.traffic_multipliers[road] = event['multiplier']
                active_events.append(event)
            else:
                road = event['road']
                if road in self.blocked_roads:
                    self.blocked_roads.remove(road)
                if road in self.traffic_multipliers:
                    self.traffic_multipliers[road] = 1.0

        self.traffic_events = active_events

    def get_travel_time_with_traffic(self, start_node: Node, end_node: Node) -> float:
        try:
            path = nx.shortest_path(self.graph, start_node.id, end_node.id, weight='weight')
            total_time = 0.0

            for i in range(len(path) - 1):
                current_node = self.nodes[path[i]]
                next_node = self.nodes[path[i + 1]]

                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data and 'weight' in edge_data:
                    base_time = edge_data['weight']

                    road_coord = (current_node.x, current_node.y)
                    if road_coord in self.blocked_roads:
                        return float('inf')

                    traffic_multiplier = self.traffic_multipliers.get(road_coord, 1.0)
                    total_time += base_time * traffic_multiplier
                else:
                    return float('inf')

            return total_time
        except nx.NetworkXNoPath:
            return float('inf')

    def get_traffic_state_vector(self) -> List[float]:
        traffic_levels = []
        blocked_count = 0

        for road in self.roads:
            if road in self.blocked_roads:
                traffic_levels.append(10.0)
                blocked_count += 1
            else:
                multiplier = self.traffic_multipliers.get(road, 1.0)
                traffic_levels.append(min(multiplier, 5.0))

        avg_traffic = sum(traffic_levels) / len(traffic_levels) if traffic_levels else 1.0
        return [avg_traffic, float(blocked_count), float(len(self.traffic_events))]