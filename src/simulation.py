import simpy
import random
from typing import List
from .environment.city_graph import CityGraph, Node
from .environment.incident_generator import IncidentGenerator
from .components.vehicle import Vehicle
from .components.station import Station, Hospital
from .agents.dispatch_center import DispatchCenter
from .utils.logger import LogData


class Simulation:
    def __init__(self, simulation_time: float = 100.0, random_seed: int = 42):
        self.env = simpy.Environment()
        self.simulation_time = simulation_time
        self.time_super = simulation_time

        random.seed(random_seed)

        self.city_graph = CityGraph(width=10, height=10)
        self.log_data = LogData()
        self.dispatch_center = DispatchCenter(self.log_data)
        self.incident_generator = IncidentGenerator(lambda_val=1.0, random_seed=random_seed)

        self.vehicles = []
        self.stations = []
        self.hospitals = []

        self.setup_environment()

    def setup_environment(self):
        station_locations = [
            self.city_graph.get_node_by_id(11),
            self.city_graph.get_node_by_id(38),
            self.city_graph.get_node_by_id(61),
            self.city_graph.get_node_by_id(88)
        ]

        for i, location in enumerate(station_locations):
            if location:
                station = Station(id=i, location=location, capacity=3)
                self.stations.append(station)

        hospital_locations = [
            self.city_graph.get_node_by_id(22),
            self.city_graph.get_node_by_id(77)
        ]

        for i, location in enumerate(hospital_locations):
            if location:
                hospital = Hospital(id=i, location=location, capacity=20)
                self.hospitals.append(hospital)

        vehicle_id = 0
        for station in self.stations:
            for _ in range(2):
                vehicle = Vehicle(
                    id=vehicle_id,
                    current_location=station.location,
                    home_station=station,
                    speed=1.0
                )
                station.add_vehicle(vehicle)
                self.vehicles.append(vehicle)
                vehicle_id += 1

        self.dispatch_center.active_vehicles = self.vehicles

    def generate_incident(self):
        while True:
            yield self.env.timeout(self.incident_generator.get_next_arrival_time())
            incident = self.incident_generator.generate_incident(self.city_graph, self.env.now)
            self.dispatch_center.receive_incident(incident)

    def dispatch_process(self):
        while True:
            yield self.env.timeout(1.0)
            self.dispatch_center.dispatch_available_vehicles(self.env.now)

    def vehicle_movement_process(self):
        while True:
            yield self.env.timeout(2.0)
            for vehicle in self.vehicles:
                if vehicle.status.value == "en_route_to_incident" and vehicle.assigned_incident:
                    vehicle.arrive_at_incident()
                    vehicle.complete_incident()
                    yield self.env.timeout(5.0)
                    self.dispatch_center.resolve_incident(vehicle.assigned_incident, self.env.now)
                    vehicle.return_to_station()
                    vehicle.arrive_at_station()

    def run_simulation(self):
        self.env.process(self.generate_incident())
        self.env.process(self.dispatch_process())
        self.env.process(self.vehicle_movement_process())

        self.env.run(until=self.simulation_time)
        self.log_summary_data()

    def get_system_state(self):
        return self.dispatch_center.get_system_state()

    def update_all_components(self):
        pass

    def log_summary_data(self):
        metrics = self.log_data.get_performance_metrics()
        print(f"Simulation completed. Summary:")
        print(f"Total incidents: {metrics.get('total_incidents', 0)}")
        print(f"Total dispatches: {metrics.get('total_dispatches', 0)}")
        print(f"Average response time: {metrics.get('avg_response_time', 0):.2f}")
        print(f"Total events logged: {metrics.get('total_events', 0)}")

        self.log_data.save_to_json("simulation_results.json")
        self.log_data.export_to_csv("simulation_results.csv")