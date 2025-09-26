from typing import List, Dict, Any, Optional
from queue import Queue
from ..components.vehicle import Vehicle, VehicleStatus
from ..components.incident import Incident, IncidentStatus
from ..utils.logger import LogData


class DispatchCenter:
    def __init__(self, log_data: LogData):
        self.pending_incidents = Queue()
        self.active_vehicles = []
        self.log_data = log_data
        self.performance_metrics = {}

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
        vehicle.dispatch_to_incident(incident, incident.location)
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
                "incident_location_y": incident.location.y
            },
            current_time
        )

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