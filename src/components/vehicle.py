from enum import Enum
from typing import Optional


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

    def dispatch_to_incident(self, incident, incident_location):
        self.status = VehicleStatus.EN_ROUTE_TO_INCIDENT
        self.assigned_incident = incident
        incident.assigned_vehicle = self

    def go_to_hospital(self, hospital_location, patient):
        self.status = VehicleStatus.TRANSPORTING_TO_HOSPITAL
        self.patient = patient

    def update_location(self, new_location):
        self.current_location = new_location

    def return_to_station(self):
        self.status = VehicleStatus.RETURNING_TO_STATION
        self.assigned_incident = None
        self.patient = None

    def get_current_location(self):
        return self.current_location

    def arrive_at_incident(self):
        self.status = VehicleStatus.AT_INCIDENT

    def complete_incident(self):
        if self.assigned_incident:
            self.assigned_incident.status = "ON_SCENE"

    def arrive_at_station(self):
        self.status = VehicleStatus.IDLE
        self.current_location = self.home_station.location