from typing import List, Dict, Any, Optional, Tuple
from queue import Queue
from ..components.vehicle import Vehicle, VehicleStatus
from ..components.incident import Incident, IncidentStatus
from ..utils.logger import LogData


class DispatchCenter:
    def __init__(self, log_data: LogData, city_graph=None):
        self.pending_incidents = Queue()
        self.active_vehicles = []
        self.log_data = log_data
        self.performance_metrics = {}
        self.city_graph = city_graph

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
        min_distance = float('inf')

        for vehicle in vehicles:
            distance = vehicle.current_location.get_distance_to(incident.location)
            if distance < min_distance:
                min_distance = distance
                best_vehicle = vehicle

        return best_vehicle

    def _dispatch_vehicle(self, vehicle: Vehicle, incident: Incident, current_time: float):
        start_location = vehicle.current_location
        if not self.city_graph.is_road(start_location.x, start_location.y):
            start_location = self.city_graph.get_nearest_road_to_building(start_location)
            vehicle.update_location(start_location)

        start_node_id = self._get_node_id(start_location)
        end_node_id = self._get_node_id(incident.location)

        path, travel_time = self._calculate_shortest_path(start_node_id, end_node_id)

        vehicle.dispatch_to_incident(incident, incident.location, path, travel_time)
        incident.assign_vehicle(vehicle)
        incident.update_status(IncidentStatus.EN_ROUTE)

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
            return self.city_graph.get_shortest_path(start_id, end_id)
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
            next_node_id = vehicle.move_along_path(current_time)
            if next_node_id is not None:
                new_location = self.city_graph.get_node_by_id(next_node_id)
                if new_location:
                    vehicle.update_location(new_location)

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
            next_node_id = vehicle.move_along_path(current_time)
            if next_node_id is not None:
                new_location = self.city_graph.get_node_by_id(next_node_id)
                if new_location:
                    vehicle.update_location(new_location)

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
            next_node_id = vehicle.move_along_path(current_time)
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

    def resolve_incident(self, incident: Incident, current_time: float):
        incident.resolve(current_time)
        response_time = incident.calculate_response_time(current_time)

        self.log_data.log_event(
            "incident_resolved",
            {
                "incident_id": incident.id,
                "resolution_time": current_time,
                "response_time": response_time,
                "vehicle_id": incident.assigned_vehicle.id if incident.assigned_vehicle else None
            },
            current_time
        )