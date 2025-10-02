from typing import List, Dict, Any, Optional, Tuple
from queue import Queue
import numpy as np
from ..components.vehicle import Vehicle, VehicleStatus
from ..components.incident import Incident, IncidentStatus
from ..utils.logger import LogData
from .dispatch_agent import DispatchAgent


class DispatchCenter:
    def __init__(self, log_data: LogData, city_graph=None, dispatch_mode: str = "heuristic",
                 shift_mode: bool = False):
        self.pending_incidents = Queue()
        self.active_vehicles = []
        self.log_data = log_data
        self.performance_metrics = {}
        self.city_graph = city_graph
        self.dispatch_mode = dispatch_mode
        self.shift_mode = shift_mode

        if dispatch_mode == "rl":
            state_dim = 10 * 4 + 5 * 4 + 3 + 3
            action_dim = 10
            self.dispatch_agent = DispatchAgent(state_dim, action_dim, shift_mode=shift_mode)
            self.training_mode = True
            self.last_state = None
            self.last_action = None
            self.episode_rewards = []
        else:
            self.dispatch_agent = None

    def receive_incident(self, incident: Incident):
        self.pending_incidents.put(incident)
        self.log_data.log_event(
            "incident_received",
            {
                "incident_id": incident.id,
                "location_x": incident.location.x,
                "location_y": incident.location.y
            },
            incident.time_created
        )

    def dispatch_available_vehicles(self, current_time: float):
        while not self.pending_incidents.empty():
            incident = self.pending_incidents.get()
            available_vehicles = self.get_available_vehicles()

            if available_vehicles:
                if self.dispatch_mode == "rl":
                    best_vehicle = self._select_rl_vehicle(incident, available_vehicles, current_time)
                else:
                    best_vehicle = self._select_closest_vehicle(incident, available_vehicles)

                if best_vehicle:
                    self._dispatch_vehicle(best_vehicle, incident, current_time)
            else:
                self.pending_incidents.put(incident)
                break

    def _select_closest_vehicle(self, incident: Incident, vehicles: List[Vehicle]) -> Optional[Vehicle]:
        if not vehicles:
            return None

        best_vehicle = None
        min_travel_time = float('inf')

        for vehicle in vehicles:
            if self.city_graph:
                travel_time = self.city_graph.get_travel_time_with_traffic(
                    vehicle.current_location, incident.location
                )
                if travel_time == float('inf'):
                    travel_time = vehicle.current_location.get_distance_to(incident.location)
            else:
                travel_time = vehicle.current_location.get_distance_to(incident.location)

            if travel_time < min_travel_time:
                min_travel_time = travel_time
                best_vehicle = vehicle

        return best_vehicle

    def _select_rl_vehicle(self, incident: Incident, available_vehicles: List[Vehicle], current_time: float) -> Optional[Vehicle]:
        if not available_vehicles or not self.dispatch_agent:
            return self._select_closest_vehicle(incident, available_vehicles)

        pending_incidents = [incident]
        while not self.pending_incidents.empty():
            try:
                pending_incidents.append(self.pending_incidents.get_nowait())
            except:
                break

        for pending_incident in pending_incidents[1:]:
            self.pending_incidents.put(pending_incident)

        current_state = self.dispatch_agent.encode_state(
            self.active_vehicles, pending_incidents, self.city_graph, current_time
        )

        if self.last_state is not None and self.last_action is not None:
            reward = self._calculate_training_reward(current_time)
            self.dispatch_agent.train_step(
                self.last_state, self.last_action, reward, current_state, False
            )

        action = self.dispatch_agent.get_action(
            current_state, self.active_vehicles, pending_incidents, self.training_mode
        )

        self.last_state = current_state
        self.last_action = action

        if action < len(self.active_vehicles):
            selected_vehicle = self.active_vehicles[action]
            if selected_vehicle in available_vehicles:
                return selected_vehicle

        return self._select_closest_vehicle(incident, available_vehicles)

    def _calculate_training_reward(self, current_time: float) -> float:
        if not hasattr(self, '_last_dispatch_time'):
            self._last_dispatch_time = current_time
            return 0.0

        time_diff = current_time - self._last_dispatch_time
        self._last_dispatch_time = current_time

        return -time_diff / 60.0

    def get_system_state(self) -> Dict[str, Any]:
        pending_count = self.pending_incidents.qsize()
        available_vehicles = self.get_available_vehicles()
        busy_vehicles = [v for v in self.active_vehicles if v.status != VehicleStatus.IDLE]

        return {
            'pending_incidents': pending_count,
            'available_vehicles': len(available_vehicles),
            'busy_vehicles': len(busy_vehicles),
            'vehicle_utilization': len(busy_vehicles) / len(self.active_vehicles) if self.active_vehicles else 0,
            'total_vehicles': len(self.active_vehicles)
        }

    def resolve_incident(self, incident: Incident, current_time: float):
        print(f"RESOLVE_INCIDENT CALLED for incident {incident.id if incident else 'None'}")
        print(f"  dispatch_mode: {self.dispatch_mode}, has dispatch_agent: {self.dispatch_agent is not None}")

        if incident and incident.assigned_vehicle:
            response_time = current_time - incident.time_created

            print(f"Incident {incident.id}: created at {incident.time_created:.1f}s, resolved at {current_time:.1f}s, response time: {response_time:.1f}s")

            if self.dispatch_mode == "rl" and self.dispatch_agent:
                print(f"  Calculating RL reward for dispatch_mode: {self.dispatch_mode}")
                reward = self.dispatch_agent.calculate_reward(
                    incident, response_time, self.get_system_state()
                )
                self.episode_rewards.append(reward)
                print(f"  RL reward: {reward:.3f}, total rewards: {len(self.episode_rewards)}")

            incident.resolution_time = current_time
            incident.status = IncidentStatus.RESOLVED

            self.log_data.log_event(
                "incident_resolved",
                {
                    "incident_id": incident.id,
                    "vehicle_id": incident.assigned_vehicle.id,
                    "response_time": response_time,
                    "resolution_time": current_time
                },
                current_time
            )

            self._last_dispatch_time = current_time
        else:
            print(f"  SKIPPED: incident={incident is not None}, assigned_vehicle={incident.assigned_vehicle if incident else 'N/A'}")

    def _dispatch_vehicle(self, vehicle: Vehicle, incident: Incident, current_time: float):
        start_location = vehicle.current_location
        if not self.city_graph.is_road(start_location.x, start_location.y):
            start_location = self.city_graph.get_nearest_road_to_building(start_location)
            vehicle.update_location(start_location)

        start_node_id = self._get_node_id(start_location)
        end_node_id = self._get_node_id(incident.location)

        path, travel_time = self._calculate_shortest_path(start_node_id, end_node_id)

        if not path:
            print(f"ERROR: No path found for vehicle {vehicle.id} from ({vehicle.current_location.x},{vehicle.current_location.y}) to incident at ({incident.location.x},{incident.location.y}) at time {current_time:.1f}")
            return

        vehicle.dispatch_to_incident(incident, incident.location, path, travel_time)
        incident.assign_vehicle(vehicle)
        incident.update_status(IncidentStatus.EN_ROUTE)

        print(f"Dispatched vehicle {vehicle.id} to incident {incident.id} at time {current_time:.1f}")
        print(f"  Vehicle location: ({vehicle.current_location.x}, {vehicle.current_location.y})")
        print(f"  Incident location: ({incident.location.x}, {incident.location.y})")
        print(f"  Path length: {len(path)}")
        print(f"  Travel time estimate: {travel_time:.1f}s")

        self.log_data.log_event(
            "vehicle_dispatched",
            {
                "vehicle_id": vehicle.id,
                "incident_id": incident.id,
                "dispatch_time": current_time,
                "vehicle_location_x": vehicle.current_location.x,
                "vehicle_location_y": vehicle.current_location.y,
                "incident_location_x": incident.location.x,
                "incident_location_y": incident.location.y,
                "estimated_travel_time": travel_time,
                "path_length": len(path)
            },
            current_time
        )

    def _get_node_id(self, location):
        return location.y * self.city_graph.width + location.x

    def _calculate_shortest_path(self, start_id: int, end_id: int) -> Tuple[List[int], float]:
        if self.city_graph:
            path, travel_time = self.city_graph.get_traffic_aware_path(start_id, end_id)
            return path, travel_time
        return [], 0.0

    def move_vehicles(self, current_time: float):
        for vehicle in self.active_vehicles:
            if vehicle.status == VehicleStatus.EN_ROUTE_TO_INCIDENT:
                self._move_vehicle_to_incident(vehicle, current_time)
            elif vehicle.status == VehicleStatus.TRANSPORTING_TO_HOSPITAL:
                self._move_vehicle_to_hospital(vehicle, current_time)
            elif vehicle.status == VehicleStatus.RETURNING_TO_STATION:
                self._move_vehicle_to_station(vehicle, current_time)

    def _move_vehicle_to_incident(self, vehicle: Vehicle, current_time: float):
        if vehicle.is_at_destination():
            print(f"Vehicle {vehicle.id} arrived at incident {vehicle.assigned_incident.id} at time {current_time:.1f}")
            vehicle.arrive_at_incident()
            self.log_data.log_event(
                "vehicle_arrived_at_incident",
                {
                    "vehicle_id": vehicle.id,
                    "incident_id": vehicle.assigned_incident.id,
                    "arrival_time": current_time
                },
                current_time
            )
        else:
            next_node_id = vehicle.move_along_path(current_time, self.city_graph)
            if next_node_id is not None:
                new_location = self.city_graph.get_node_by_id(next_node_id)
                if new_location:
                    vehicle.update_location(new_location)
                else:
                    print(f"ERROR: Vehicle {vehicle.id} got invalid node_id {next_node_id} at time {current_time:.1f}")
            else:
                if not vehicle.is_at_destination():
                    print(f"STUCK: Vehicle {vehicle.id} can't move but not at destination. Pos: ({vehicle.current_location.x}, {vehicle.current_location.y}), Target: ({vehicle.destination.x}, {vehicle.destination.y}) at time {current_time:.1f}")

    def _move_vehicle_to_hospital(self, vehicle: Vehicle, current_time: float):
        if vehicle.is_at_destination():
            nearest_road = self.city_graph.get_nearest_road_to_building(vehicle.home_station.location)
            if nearest_road:
                hospital_path, hospital_time = self._calculate_shortest_path(
                    self._get_node_id(vehicle.current_location),
                    self._get_node_id(nearest_road)
                )
                vehicle.return_to_station(hospital_path, hospital_time)
                vehicle.destination = nearest_road
                self.log_data.log_event(
                    "patient_delivered_to_hospital",
                    {
                        "vehicle_id": vehicle.id,
                        "delivery_time": current_time
                    },
                    current_time
                )
        else:
            next_node_id = vehicle.move_along_path(current_time, self.city_graph)
            if next_node_id is not None:
                new_location = self.city_graph.get_node_by_id(next_node_id)
                if new_location:
                    vehicle.update_location(new_location)
                else:
                    print(f"ERROR: Vehicle {vehicle.id} (hospital) got invalid node_id {next_node_id} at time {current_time:.1f}")
            else:
                if not vehicle.is_at_destination():
                    print(f"STUCK: Vehicle {vehicle.id} (hospital) can't move. Pos: ({vehicle.current_location.x}, {vehicle.current_location.y}), Target: ({vehicle.destination.x}, {vehicle.destination.y}) at time {current_time:.1f}")

    def _move_vehicle_to_station(self, vehicle: Vehicle, current_time: float):
        if vehicle.is_at_destination():
            vehicle.update_location(vehicle.home_station.location)
            vehicle.arrive_at_station()
            self.log_data.log_event(
                "vehicle_returned_to_station",
                {
                    "vehicle_id": vehicle.id,
                    "return_time": current_time
                },
                current_time
            )
        else:
            next_node_id = vehicle.move_along_path(current_time, self.city_graph)
            if next_node_id is not None:
                new_location = self.city_graph.get_node_by_id(next_node_id)
                if new_location:
                    vehicle.update_location(new_location)

    def get_available_vehicles(self) -> List[Vehicle]:
        return [v for v in self.active_vehicles if v.status == VehicleStatus.IDLE]

    def get_system_state(self) -> Dict[str, Any]:
        state = {
            "pending_incidents": self.pending_incidents.qsize(),
            "available_vehicles": len(self.get_available_vehicles()),
            "total_vehicles": len(self.active_vehicles),
            "vehicles": []
        }

        for vehicle in self.active_vehicles:
            vehicle_state = {
                "id": vehicle.id,
                "status": vehicle.status.value,
                "location_x": vehicle.current_location.x,
                "location_y": vehicle.current_location.y
            }
            state["vehicles"].append(vehicle_state)

        return state

    def update_vehicle_status(self, vehicle: Vehicle, new_status: VehicleStatus):
        vehicle.status = new_status

