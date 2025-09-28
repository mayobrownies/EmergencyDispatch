import simpy
import random
from typing import List
from .environment.city_graph import CityGraph, Node
from .environment.incident_generator import IncidentGenerator
from .components.vehicle import Vehicle, VehicleStatus
from .components.station import Station, Hospital
from .agents.dispatch_center import DispatchCenter
from .utils.logger import LogData


class Simulation:
    def __init__(self, simulation_time: float = 100.0, random_seed: int = 42, dispatch_mode: str = "heuristic"):
        self.env = simpy.Environment()
        self.simulation_time = simulation_time
        self.time_super = simulation_time
        self.dispatch_mode = dispatch_mode

        random.seed(random_seed)

        self.city_graph = CityGraph(width=20, height=15)
        self.log_data = LogData()
        self.dispatch_center = DispatchCenter(self.log_data, self.city_graph, dispatch_mode)
        self.incident_generator = IncidentGenerator(lambda_val=0.05, random_seed=random_seed)

        self.vehicles = []
        self.stations = []
        self.hospitals = []

        self.setup_environment()

    def setup_environment(self):
        road_nodes = self.city_graph.get_road_nodes()

        station_positions = [
            (1, 1), (18, 2), (2, 13), (17, 8)
        ]

        for i, (x, y) in enumerate(station_positions):
            node_id = y * self.city_graph.width + x
            location = self.city_graph.get_node_by_id(node_id)
            if location and not self.city_graph.is_road(x, y):
                station = Station(id=i, location=location, capacity=3)
                self.stations.append(station)
                self.city_graph.buildings.add((x, y))

        hospital_positions = [
            (10, 1), (6, 10)
        ]

        for i, (x, y) in enumerate(hospital_positions):
            node_id = y * self.city_graph.width + x
            location = self.city_graph.get_node_by_id(node_id)
            if location and not self.city_graph.is_road(x, y):
                hospital = Hospital(id=i, location=location, capacity=20)
                self.hospitals.append(hospital)
                self.city_graph.buildings.add((x, y))

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
            yield self.env.timeout(1.0)
            self.dispatch_center.move_vehicles(self.env.now)

    def incident_resolution_process(self):
        vehicle_timers = {}

        while True:
            yield self.env.timeout(1.0)

            for vehicle in self.vehicles:
                if vehicle.status == VehicleStatus.AT_INCIDENT and vehicle.assigned_incident:
                    if vehicle.id not in vehicle_timers:
                        vehicle_timers[vehicle.id] = self.env.now + 5.0

                    if self.env.now >= vehicle_timers[vehicle.id]:
                        nearest_hospital = self.city_graph.get_nearest_hospital(
                            vehicle.current_location, self.hospitals
                        )

                        if nearest_hospital:
                            nearest_road = self.city_graph.get_nearest_road_to_building(nearest_hospital.location)
                            if nearest_road:
                                hospital_path, hospital_time = self.city_graph.get_shortest_path(
                                    self.dispatch_center._get_node_id(vehicle.current_location),
                                    self.dispatch_center._get_node_id(nearest_road)
                                )
                                vehicle.go_to_hospital(nearest_road, "patient", hospital_path, hospital_time)

                        self.dispatch_center.resolve_incident(vehicle.assigned_incident, self.env.now)
                        del vehicle_timers[vehicle.id]

    def run_simulation(self):
        self.env.process(self.generate_incident())
        self.env.process(self.dispatch_process())
        self.env.process(self.vehicle_movement_process())
        self.env.process(self.incident_resolution_process())

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

        if self.dispatch_mode == "rl" and self.dispatch_center.dispatch_agent:
            avg_reward = sum(self.dispatch_center.episode_rewards) / len(self.dispatch_center.episode_rewards) if self.dispatch_center.episode_rewards else 0
            print(f"Average episode reward: {avg_reward:.2f}")
            print(f"Current epsilon: {self.dispatch_center.dispatch_agent.epsilon:.3f}")

        self.log_data.save_to_json("simulation_results.json")
        self.log_data.export_to_csv("simulation_results.csv")

    def reset_for_training(self):
        """Reset simulation state for training episodes."""
        self.env = simpy.Environment()
        self.log_data = LogData()

        for vehicle in self.vehicles:
            vehicle.status = VehicleStatus.IDLE
            vehicle.current_location = vehicle.home_station.location
            vehicle.assigned_incident = None
            vehicle.patient = None
            vehicle.current_path = []
            vehicle.path_index = 0

        while not self.dispatch_center.pending_incidents.empty():
            try:
                self.dispatch_center.pending_incidents.get_nowait()
            except:
                break

        if self.dispatch_mode == "rl":
            self.dispatch_center.episode_rewards = []
            self.dispatch_center.last_state = None
            self.dispatch_center.last_action = None

    def run_training_episodes(self, num_episodes: int = 100, episode_length: float = 300.0):
        """Run multiple training episodes for RL agent."""
        if self.dispatch_mode != "rl":
            print("Training only available in RL mode")
            return

        episode_rewards = []

        for episode in range(num_episodes):
            print(f"Training episode {episode + 1}/{num_episodes}")

            self.reset_for_training()
            self.simulation_time = episode_length
            self.time_super = episode_length

            try:
                self.run_simulation()
            except Exception as e:
                print(f"Episode {episode + 1} failed: {e}")
                continue

            episode_reward = sum(self.dispatch_center.episode_rewards) if self.dispatch_center.episode_rewards else 0
            episode_rewards.append(episode_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(f"Episodes {episode - 8}-{episode + 1} avg reward: {avg_reward:.2f}")

                if (episode + 1) % 50 == 0:
                    self.dispatch_center.dispatch_agent.save_model(f"models/dqn_episode_{episode + 1}.pth")

        final_avg = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        print(f"Training completed. Final average reward: {final_avg:.2f}")

        self.dispatch_center.dispatch_agent.save_model("models/dqn_final.pth")
        return episode_rewards