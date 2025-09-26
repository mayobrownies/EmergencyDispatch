from enum import Enum
from typing import Optional


class IncidentStatus(Enum):
    UNASSIGNED = "unassigned"
    AWAITING_VEHICLE = "awaiting_vehicle"
    EN_ROUTE = "en_route"
    ON_SCENE = "on_scene"
    TRANSPORTING = "transporting"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class Incident:
    def __init__(self, id: int, location, time_created: float):
        self.id = id
        self.location = location
        self.time_created = time_created
        self.status = IncidentStatus.UNASSIGNED
        self.assigned_vehicle = None
        self.resolution_time = None
        self.response_time = None

    def update_status(self, new_status: IncidentStatus):
        self.status = new_status

    def calculate_response_time(self, arrival_time: float) -> float:
        self.response_time = arrival_time - self.time_created
        return self.response_time

    def assign_vehicle(self, vehicle):
        self.assigned_vehicle = vehicle
        self.status = IncidentStatus.AWAITING_VEHICLE

    def get_resolution_time(self) -> Optional[float]:
        if self.resolution_time and self.time_created:
            return self.resolution_time - self.time_created
        return None

    def resolve(self, resolution_time: float):
        self.resolution_time = resolution_time
        self.status = IncidentStatus.RESOLVED