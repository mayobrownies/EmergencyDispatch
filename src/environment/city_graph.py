import networkx as nx
import osmnx as ox
import random
import numpy as np
from typing import List, Tuple, Optional
import pickle
import os
from config import TrafficConfig


class Node:
    def __init__(self, id: int, x: float, y: float, osmid: Optional[int] = None):
        self.id = id
        self.x = x
        self.y = y
        self.osmid = osmid

    def get_coords(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_distance_to(self, other_node: 'Node') -> float:
        return np.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)


class CityGraph:
    def __init__(self, place_name: str = "Midtown, Atlanta, Georgia, USA", use_cache: bool = True):
        self.graph = nx.Graph()
        self.nodes = []
        self.place_name = place_name
        self.traffic_multipliers = {}
        self.blocked_roads = set()
        self.traffic_events = []

        self.min_lat = None
        self.max_lat = None
        self.min_lon = None
        self.max_lon = None

        self.width = 1000
        self.height = 1000

        self.roads = set()
        self.buildings = set()

        self._load_or_download_graph(use_cache)
        self._initialize_traffic_system()

    def _load_or_download_graph(self, use_cache: bool):
        cache_file = "data/atlanta_graph.pkl"

        if use_cache and os.path.exists(cache_file):
            print(f"Loading cached graph from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.osm_graph = cached_data['graph']
                self.min_lat = cached_data['min_lat']
                self.max_lat = cached_data['max_lat']
                self.min_lon = cached_data['min_lon']
                self.max_lon = cached_data['max_lon']
        else:
            print(f"Downloading street network for Atlanta Midtown area...")
            self.osm_graph = ox.graph_from_point(
                (33.7756, -84.3963),
                dist=1500,
                network_type='drive',
                simplify=True
            )

            lats = [data['y'] for node, data in self.osm_graph.nodes(data=True)]
            lons = [data['x'] for node, data in self.osm_graph.nodes(data=True)]
            self.min_lat = min(lats)
            self.max_lat = max(lats)
            self.min_lon = min(lons)
            self.max_lon = max(lons)

            os.makedirs("data", exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'graph': self.osm_graph,
                    'min_lat': self.min_lat,
                    'max_lat': self.max_lat,
                    'min_lon': self.min_lon,
                    'max_lon': self.max_lon
                }, f)
            print(f"Cached graph to {cache_file}")

        self._convert_osm_to_simulation_graph()

    def _convert_osm_to_simulation_graph(self):
        osm_to_sim_id = {}
        node_id = 0

        for osm_id, data in self.osm_graph.nodes(data=True):
            lat = data['y']
            lon = data['x']

            x = self._lon_to_x(lon)
            y = self._lat_to_y(lat)

            node = Node(node_id, x, y, osmid=osm_id)
            self.nodes.append(node)
            self.roads.add((x, y))

            self.graph.add_node(node_id, pos=(x, y), node_obj=node, osmid=osm_id)
            osm_to_sim_id[osm_id] = node_id
            node_id += 1

        for u, v, data in self.osm_graph.edges(data=True):
            sim_u = osm_to_sim_id[u]
            sim_v = osm_to_sim_id[v]

            length = data.get('length', 100.0)
            speed_kph = data.get('maxspeed', 40)

            if isinstance(speed_kph, list):
                speed_str = speed_kph[0] if speed_kph else '40'
            elif isinstance(speed_kph, str):
                speed_str = speed_kph
            else:
                speed_str = str(speed_kph) if speed_kph else '40'

            try:
                speed_val = float(speed_str.split()[0])
                if 'mph' in speed_str.lower():
                    speed_kph = speed_val * 1.60934
                else:
                    speed_kph = speed_val
            except:
                speed_kph = 40.0

            travel_time = (length / 1000.0) / speed_kph * 3600.0

            self.graph.add_edge(sim_u, sim_v, weight=travel_time, length=length, speed=speed_kph)

    def _lat_to_y(self, lat: float) -> float:
        return ((lat - self.min_lat) / (self.max_lat - self.min_lat)) * self.height

    def _lon_to_x(self, lon: float) -> float:
        return ((lon - self.min_lon) / (self.max_lon - self.min_lon)) * self.width

    def _y_to_lat(self, y: float) -> float:
        return self.min_lat + (y / self.height) * (self.max_lat - self.min_lat)

    def _x_to_lon(self, x: float) -> float:
        return self.min_lon + (x / self.width) * (self.max_lon - self.min_lon)

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
                    if node and (int(node.x), int(node.y)) == road_coord:
                        edges_to_remove = list(temp_graph.edges(node_id))
                        temp_graph.remove_edges_from(edges_to_remove)

            path = nx.shortest_path(temp_graph, start_node, end_node, weight='weight')

            total_time = 0.0
            for i in range(len(path) - 1):
                current_node = self.nodes[path[i]]
                edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                if edge_data and 'weight' in edge_data:
                    base_time = edge_data['weight']
                    road_coord = (int(current_node.x), int(current_node.y))
                    traffic_multiplier = self.traffic_multipliers.get(road_coord, 1.0)
                    total_time += base_time * traffic_multiplier
                else:
                    return [], float('inf')

            return path, total_time
        except nx.NetworkXNoPath:
            return [], float('inf')

    def get_random_node(self) -> Node:
        return random.choice(self.nodes)

    def get_nearest_road_to_building(self, building_location: Node) -> Optional[Node]:
        min_distance = float('inf')
        nearest_road = None

        for node in self.nodes:
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

    def is_road(self, x: int, y: int) -> bool:
        return (x, y) in self.roads

    def is_building(self, x: int, y: int) -> bool:
        return (x, y) in self.buildings

    def get_road_nodes(self) -> List[Node]:
        return self.nodes

    def _initialize_traffic_system(self):
        for node in self.nodes:
            road_coord = (int(node.x), int(node.y))
            self.traffic_multipliers[road_coord] = 1.0

    def reset_traffic(self):
        self.traffic_multipliers.clear()
        self.blocked_roads.clear()
        self.traffic_events.clear()
        self._initialize_traffic_system()

    def update_traffic_conditions(self, current_time: float):
        self._update_dynamic_traffic(current_time)
        self._process_traffic_events(current_time)

    def _update_dynamic_traffic(self, current_time: float):
        rush_hour_multiplier = self._get_rush_hour_multiplier(current_time)

        for node in self.nodes:
            road_coord = (int(node.x), int(node.y))
            if road_coord not in self.blocked_roads:
                current_multiplier = self.traffic_multipliers.get(road_coord, 1.0)
                decay_rate = TrafficConfig.DECAY_RATE
                self.traffic_multipliers[road_coord] = current_multiplier * (1 - decay_rate) + TrafficConfig.DECAY_TARGET * decay_rate

        num_roads_to_update = max(10, len(self.nodes) // 50)
        roads_to_update = random.sample(self.nodes, min(num_roads_to_update, len(self.nodes)))

        for node in roads_to_update:
            road_coord = (int(node.x), int(node.y))
            if road_coord not in self.blocked_roads:
                num_edges = len(list(self.graph.edges(node.id)))

                is_major_road = num_edges >= 3

                if is_major_road:
                    base_multiplier = (rush_hour_multiplier - 1.0) * TrafficConfig.MAJOR_ROAD_MULTIPLIER + 1.0
                    variation = random.uniform(*TrafficConfig.MAJOR_ROAD_VARIATION)
                else:
                    base_multiplier = (rush_hour_multiplier - 1.0) * TrafficConfig.MINOR_ROAD_MULTIPLIER + 1.0
                    variation = random.uniform(*TrafficConfig.MINOR_ROAD_VARIATION)

                target_multiplier = base_multiplier * variation
                target_multiplier = max(0.7, min(target_multiplier, 3.0))

                self.traffic_multipliers[road_coord] = target_multiplier

    def _get_rush_hour_multiplier(self, current_time: float) -> float:
        hour = (current_time / 3600.0) % 24

        if 7.5 <= hour <= 8.5 or 17.5 <= hour <= 18.5:
            return 2.2
        elif 6.5 <= hour < 7.5 or 8.5 < hour <= 9.5 or 16.5 <= hour < 17.5 or 18.5 < hour <= 19.5:
            return 1.6
        elif 6 <= hour < 6.5 or 9.5 < hour <= 10 or 16 <= hour < 16.5 or 19.5 < hour <= 20:
            return 1.3
        elif 11 <= hour <= 15:
            return 1.1
        else:
            return 0.8

    def add_traffic_incident(self, duration: float, severity: str = "moderate", start_time: float = 0, rng=None):
        if self.nodes:
            if rng is None:
                rng = random
            affected_node = rng.choice(self.nodes)
            affected_road = (int(affected_node.x), int(affected_node.y))

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
                if event['severity'] == "severe" and not event.get('blocked_applied', False):
                    self.blocked_roads.add(road)
                    event['blocked_applied'] = True
                elif event['severity'] != "severe":
                    self.traffic_multipliers[road] = event['multiplier']
                active_events.append(event)
            else:
                road = event['road']
                if road in self.blocked_roads:
                    self.blocked_roads.remove(road)
                    print(f"Road blockage cleared at {road} at time {current_time:.1f}")
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

                    road_coord = (int(current_node.x), int(current_node.y))
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

        for node in self.nodes:
            road_coord = (int(node.x), int(node.y))
            if road_coord in self.blocked_roads:
                traffic_levels.append(10.0)
                blocked_count += 1
            else:
                multiplier = self.traffic_multipliers.get(road_coord, 1.0)
                traffic_levels.append(min(multiplier, 5.0))

        avg_traffic = sum(traffic_levels) / len(traffic_levels) if traffic_levels else 1.0
        normalized_avg_traffic = avg_traffic / 5.0
        normalized_blocked = blocked_count / max(1, len(self.nodes))
        normalized_events = len(self.traffic_events) / 10.0

        return [normalized_avg_traffic, normalized_blocked, normalized_events]
