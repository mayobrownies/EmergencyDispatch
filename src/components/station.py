from typing import List
from .vehicle import Vehicle, VehicleStatus


class Station:
    def __init__(self, id: int, location, capacity: int = 5):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.stationed_vehicles = []

    def add_vehicle(self, vehicle: Vehicle):
        if len(self.stationed_vehicles) < self.capacity:
            self.stationed_vehicles.append(vehicle)
            vehicle.home_station = self
            vehicle.current_location = self.location
            return True
        return False

    def remove_vehicle(self, vehicle: Vehicle):
        if vehicle in self.stationed_vehicles:
            self.stationed_vehicles.remove(vehicle)
            return True
        return False

    def get_available_vehicles(self) -> List[Vehicle]:
        return [v for v in self.stationed_vehicles if v.status == VehicleStatus.IDLE]

    def get_occupancy_rate(self) -> float:
        return len(self.stationed_vehicles) / self.capacity if self.capacity > 0 else 0.0


class Hospital:
    def __init__(self, id: int, location, capacity: int = 50):
        self.id = id
        self.location = location
        self.capacity = capacity
        self.current_patients = 0

    def admit_patient(self, vehicle):
        if self.current_patients < self.capacity:
            self.current_patients += 1
            vehicle.patient = None
            return True
        return False

    def get_occupancy_rate(self) -> float:
        return self.current_patients / self.capacity if self.capacity > 0 else 0.0

    def discharge_patient(self):
        if self.current_patients > 0:
            self.current_patients -= 1