import random
import numpy as np
from typing import Optional
from ..components.incident import Incident


class IncidentGenerator:
    def __init__(self, lambda_val: float = 2.0, random_seed: Optional[int] = None):
        self.lambda_val = lambda_val
        self.random_seed = random_seed
        self.incident_counter = 0

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def get_next_arrival_time(self) -> float:
        return np.random.exponential(1.0 / self.lambda_val)

    def generate_incident(self, city_graph, current_time: float) -> Incident:
        location = city_graph.get_random_node()
        incident = Incident(
            id=self.incident_counter,
            location=location,
            time_created=current_time
        )
        self.incident_counter += 1
        return incident

    def create_incident(self, location, current_time: float) -> Incident:
        incident = Incident(
            id=self.incident_counter,
            location=location,
            time_created=current_time
        )
        self.incident_counter += 1
        return incident