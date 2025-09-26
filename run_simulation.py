import pygame
import sys
import threading
import time
from src.simulation import Simulation
from src.components.vehicle import VehicleStatus
from src.components.incident import IncidentStatus


class SimulationVisualizer:
    def __init__(self, simulation):
        pygame.init()
        self.simulation = simulation
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Emergency Dispatch Simulation")

        self.grid_width = simulation.city_graph.width
        self.grid_height = simulation.city_graph.height
        self.cell_width = self.width // self.grid_width
        self.cell_height = self.height // self.grid_height

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        self.running = True
        self.simulation_thread = None

    def draw_grid(self):
        for x in range(0, self.width, self.cell_width):
            pygame.draw.line(self.screen, self.BLACK, (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_height):
            pygame.draw.line(self.screen, self.BLACK, (0, y), (self.width, y))

    def draw_stations(self):
        for station in self.simulation.stations:
            x = station.location.x * self.cell_width + self.cell_width // 2
            y = station.location.y * self.cell_height + self.cell_height // 2
            pygame.draw.circle(self.screen, self.BLUE, (x, y), 15)

            text = self.small_font.render(f"S{station.id}", True, self.WHITE)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def draw_hospitals(self):
        for hospital in self.simulation.hospitals:
            x = hospital.location.x * self.cell_width + self.cell_width // 2
            y = hospital.location.y * self.cell_height + self.cell_height // 2
            pygame.draw.rect(self.screen, self.GREEN,
                           (x - 15, y - 10, 30, 20))

            text = self.small_font.render(f"H{hospital.id}", True, self.WHITE)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def draw_vehicles(self):
        for vehicle in self.simulation.vehicles:
            x = vehicle.current_location.x * self.cell_width + self.cell_width // 4
            y = vehicle.current_location.y * self.cell_height + self.cell_height // 4

            if vehicle.status == VehicleStatus.IDLE:
                color = self.WHITE
            elif vehicle.status == VehicleStatus.EN_ROUTE_TO_INCIDENT:
                color = self.YELLOW
            elif vehicle.status == VehicleStatus.AT_INCIDENT:
                color = self.ORANGE
            elif vehicle.status == VehicleStatus.TRANSPORTING_TO_HOSPITAL:
                color = self.PURPLE
            else:
                color = self.BLACK

            pygame.draw.circle(self.screen, color, (x, y), 8)

            text = self.small_font.render(str(vehicle.id), True, self.BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def draw_incidents(self):
        active_incidents = []
        for event in self.simulation.log_data.history:
            if event["event_type"] == "incident_received":
                incident_data = event["data"]
                resolved = False

                for resolve_event in self.simulation.log_data.history:
                    if (resolve_event["event_type"] == "incident_resolved" and
                        resolve_event["data"]["incident_id"] == incident_data["incident_id"]):
                        resolved = True
                        break

                if not resolved:
                    active_incidents.append(incident_data)

        for incident_data in active_incidents:
            x = incident_data["location_x"] * self.cell_width + 3 * self.cell_width // 4
            y = incident_data["location_y"] * self.cell_height + 3 * self.cell_height // 4
            pygame.draw.circle(self.screen, self.RED, (x, y), 6)

    def draw_stats(self):
        metrics = self.simulation.log_data.get_performance_metrics()
        stats_text = [
            f"Time: {self.simulation.env.now:.1f}",
            f"Incidents: {metrics.get('total_incidents', 0)}",
            f"Dispatches: {metrics.get('total_dispatches', 0)}",
            f"Avg Response: {metrics.get('avg_response_time', 0):.2f}",
            f"Available Vehicles: {len(self.simulation.dispatch_center.get_available_vehicles())}"
        ]

        for i, text in enumerate(stats_text):
            rendered = self.font.render(text, True, self.BLACK)
            self.screen.blit(rendered, (10, 10 + i * 25))

    def draw_legend(self):
        legend_items = [
            ("Stations", self.BLUE),
            ("Hospitals", self.GREEN),
            ("Idle Vehicles", self.WHITE),
            ("Responding", self.YELLOW),
            ("At Scene", self.ORANGE),
            ("Incidents", self.RED)
        ]

        legend_x = self.width - 150
        legend_y = 10

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 25
            pygame.draw.circle(self.screen, color, (legend_x, y_pos + 8), 8)
            text = self.small_font.render(label, True, self.BLACK)
            self.screen.blit(text, (legend_x + 20, y_pos))

    def run_simulation_thread(self):
        self.simulation.run_simulation()

    def run(self):
        self.simulation_thread = threading.Thread(target=self.run_simulation_thread)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill(self.WHITE)

            self.draw_grid()
            self.draw_stations()
            self.draw_hospitals()
            self.draw_vehicles()
            self.draw_incidents()
            self.draw_stats()
            self.draw_legend()

            pygame.display.flip()
            clock.tick(10)

        pygame.quit()


def main():
    print("Starting Emergency Dispatch Simulation...")

    simulation = Simulation(simulation_time=50.0, random_seed=42)

    if len(sys.argv) > 1 and sys.argv[1] == "--no-display":
        print("Running simulation without display...")
        simulation.run_simulation()
    else:
        print("Starting simulation with visual display...")
        print("Close the window to stop the simulation.")
        visualizer = SimulationVisualizer(simulation)
        visualizer.run()


if __name__ == "__main__":
    main()