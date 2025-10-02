import random
from enum import Enum



class VehicleStatus(Enum):
    IDLE = "idle"
    EN_ROUTE_TO_INCIDENT = "en_route_to_incident"
    AT_INCIDENT = "at_incident"
    TRANSPORTING_TO_HOSPITAL = "transporting_to_hospital"
    RETURNING_TO_STATION = "returning_to_station"
    OUT_OF_SERVICE = "out_of_service"


class Vehicle:
    def __init__(self, id: int, current_location, home_station, speed: float = 1.0):
        self.id = id
        self.current_location = current_location
        self.status = VehicleStatus.IDLE
        self.speed = speed
        self.home_station = home_station
        self.assigned_incident = None
        self.patient = None
        self.current_path = []
        self.path_index = 0
        self.destination = None
        self.arrival_time = None

    def dispatch_to_incident(self, incident, incident_location, path=None, travel_time=0):
        self.status = VehicleStatus.EN_ROUTE_TO_INCIDENT
        self.assigned_incident = incident
        self.destination = incident_location
        self.travel_time_estimate = travel_time
        if path:
            self.current_path = path
            self.path_index = 0
            self.arrival_time = travel_time
        incident.assigned_vehicle = self

    def go_to_hospital(self, hospital_location, patient, path=None, travel_time=0):
        self.status = VehicleStatus.TRANSPORTING_TO_HOSPITAL
        self.patient = patient
        self.destination = hospital_location
        if path:
            self.current_path = path
            self.path_index = 0
            self.arrival_time = travel_time

    def update_location(self, new_location):
        self.current_location = new_location

    def move_along_path(self, current_time, city_graph=None):
        if not self.current_path or self.path_index >= len(self.current_path):
            if not self.current_path:
                print(f"WARNING: Vehicle {self.id} has no path at time {current_time:.1f}")
            else:
                print(f"WARNING: Vehicle {self.id} reached end of path (index {self.path_index}/{len(self.current_path)}) at time {current_time:.1f}")
            return False

        if city_graph:
            road_coord = (self.current_location.x, self.current_location.y)

            if road_coord in city_graph.blocked_roads:
                return False

            if self.path_index < len(self.current_path) - 1:
                next_node_id = self.current_path[self.path_index + 1]
                next_node = city_graph.get_node_by_id(next_node_id)
                if next_node:
                    next_road_coord = (next_node.x, next_node.y)
                    if next_road_coord in city_graph.blocked_roads:
                        return False

            traffic_multiplier = city_graph.traffic_multipliers.get(road_coord, 1.0)

            if traffic_multiplier >= 2.0:
                move_chance = 0.2
            elif traffic_multiplier >= 1.3:
                move_chance = 0.70
            else:
                move_chance = 1.0

            random.seed(int(current_time * 10 + self.id))
            if random.random() > move_chance:
                return False

        if self.path_index < len(self.current_path) - 1:
            self.path_index += 1
            node_id = self.current_path[self.path_index]
            return node_id
        else:
            print(f"Vehicle {self.id} completed path at time {current_time:.1f}")
            return None

    def return_to_station(self, path=None, travel_time=0):
        self.status = VehicleStatus.RETURNING_TO_STATION
        self.destination = self.home_station.location
        if path:
            self.current_path = path
            self.path_index = 0
            self.arrival_time = travel_time
        self.assigned_incident = None
        self.patient = None

    def get_current_location(self):
        return self.current_location

    def arrive_at_incident(self):
        self.status = VehicleStatus.AT_INCIDENT
        self.current_path = []
        self.path_index = 0

    def complete_incident(self):
        if self.assigned_incident:
            self.assigned_incident.status = "ON_SCENE"

    def arrive_at_station(self):
        self.status = VehicleStatus.IDLE
        self.current_location = self.home_station.location
        self.current_path = []
        self.path_index = 0
        self.destination = None

    def is_at_destination(self):
        if not self.destination:
            return False
        return (self.current_location.x == self.destination.x and
                self.current_location.y == self.destination.y)