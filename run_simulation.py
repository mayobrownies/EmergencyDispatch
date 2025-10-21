import pygame
import traceback
import threading
from src.simulation import Simulation
from src.components.vehicle import VehicleStatus

class SimulationVisualizer:
    def __init__(self, simulation, dispatch_mode="heuristic"):
        pygame.init()
        self.simulation = simulation
        self.control_panel_width = 200
        self.sidebar_width = 300
        self.simulation_width = 1000
        self.width = self.control_panel_width + self.simulation_width + self.sidebar_width
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Emergency Dispatch Simulation")

        self.city_width = simulation.city_graph.width
        self.city_height = simulation.city_graph.height
        self.top_padding = 30
        self.bottom_padding = 10
        self.available_height = self.height - self.top_padding - self.bottom_padding
        self.scale_x = self.simulation_width / self.city_width
        self.scale_y = self.available_height / self.city_height

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.ORANGE = (255, 165, 0)
        self.PURPLE = (128, 0, 128)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_GRAY = (200, 200, 200)
        self.BUTTON_GRAY = (180, 180, 180)

        self.TRAFFIC_LIGHT = (0, 255, 255)
        self.TRAFFIC_MODERATE = (255, 255, 0)
        self.TRAFFIC_HEAVY = (255, 100, 0)
        self.TRAFFIC_BLOCKED = (255, 0, 0)

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)

        self.running = True
        self.simulation_thread = None
        self.auto_mode = False
        self.step_requested = False
        self.simulation_paused = True
        self.simulation_completed = False

        self.next_button = pygame.Rect(20, 50, 160, 40)
        self.auto_button = pygame.Rect(20, 100, 160, 40)
        self.speedup_button = pygame.Rect(20, 150, 160, 40)
        self.mode_button = pygame.Rect(20, 200, 160, 40)
        self.shift_button = pygame.Rect(20, 250, 160, 40)
        self.train_button = pygame.Rect(20, 300, 160, 40)
        self.reset_button = pygame.Rect(20, 350, 160, 40)

        self.speed_multiplier = 1
        self.speed_options = [1, 2, 5, 10, 50]
        self.dispatch_mode = dispatch_mode
        self.shift_mode = hasattr(simulation, 'shift_mode') and simulation.shift_mode
        self.training_active = False

    def draw_control_panel(self):
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, 0, self.control_panel_width, self.height))
        pygame.draw.line(self.screen, self.BLACK, (self.control_panel_width, 0), (self.control_panel_width, self.height), 2)

        title = self.font.render("CONTROLS", True, self.BLACK)
        self.screen.blit(title, (20, 20))

        next_color = self.BUTTON_GRAY if not self.simulation_completed else self.GRAY
        pygame.draw.rect(self.screen, next_color, self.next_button)
        pygame.draw.rect(self.screen, self.BLACK, self.next_button, 2)
        next_text = self.small_font.render("NEXT STEP", True, self.BLACK)
        self.screen.blit(next_text, (self.next_button.x + 35, self.next_button.y + 12))

        auto_color = self.GREEN if self.auto_mode else self.BUTTON_GRAY
        if self.simulation_completed:
            auto_color = self.GRAY
        pygame.draw.rect(self.screen, auto_color, self.auto_button)
        pygame.draw.rect(self.screen, self.BLACK, self.auto_button, 2)
        auto_text = self.small_font.render("AUTO MODE", True, self.BLACK)
        self.screen.blit(auto_text, (self.auto_button.x + 35, self.auto_button.y + 12))

        pygame.draw.rect(self.screen, self.BUTTON_GRAY, self.speedup_button)
        pygame.draw.rect(self.screen, self.BLACK, self.speedup_button, 2)
        speed_text = self.small_font.render(f"SPEED: {self.speed_multiplier}x", True, self.BLACK)
        self.screen.blit(speed_text, (self.speedup_button.x + 35, self.speedup_button.y + 12))

        mode_color = self.BLUE if self.dispatch_mode == "rl" else self.BUTTON_GRAY
        pygame.draw.rect(self.screen, mode_color, self.mode_button)
        pygame.draw.rect(self.screen, self.BLACK, self.mode_button, 2)
        mode_text = self.small_font.render(f"MODE: {self.dispatch_mode.upper()}", True, self.BLACK)
        self.screen.blit(mode_text, (self.mode_button.x + 25, self.mode_button.y + 12))

        shift_color = self.ORANGE if self.shift_mode else self.BUTTON_GRAY
        pygame.draw.rect(self.screen, shift_color, self.shift_button)
        pygame.draw.rect(self.screen, self.BLACK, self.shift_button, 2)
        shift_text = self.small_font.render("8-HOUR SHIFT" if self.shift_mode else "5-MIN DEMO", True, self.BLACK)
        self.screen.blit(shift_text, (self.shift_button.x + 25, self.shift_button.y + 12))

        train_color = self.GREEN if self.training_active else self.GRAY if self.dispatch_mode != "rl" else self.BUTTON_GRAY
        pygame.draw.rect(self.screen, train_color, self.train_button)
        pygame.draw.rect(self.screen, self.BLACK, self.train_button, 2)
        train_text = self.small_font.render("TRAINING" if self.training_active else "START TRAIN", True, self.BLACK)
        self.screen.blit(train_text, (self.train_button.x + 35, self.train_button.y + 12))

        pygame.draw.rect(self.screen, self.BUTTON_GRAY, self.reset_button)
        pygame.draw.rect(self.screen, self.BLACK, self.reset_button, 2)
        reset_text = self.small_font.render("RESET", True, self.BLACK)
        self.screen.blit(reset_text, (self.reset_button.x + 55, self.reset_button.y + 12))

        status_text = "AUTO RUNNING" if self.auto_mode else "PAUSED" if self.simulation_paused else "MANUAL"
        if self.simulation_completed:
            status_text = "COMPLETED"
        status_color = self.GREEN if self.auto_mode else self.ORANGE if self.simulation_paused else self.BLUE
        if self.simulation_completed:
            status_color = self.RED

        status_label = self.small_font.render("Status:", True, self.BLACK)
        self.screen.blit(status_label, (20, 410))
        status_display = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status_display, (20, 430))

        time_label = self.small_font.render(f"Time: {self.simulation.env.now:.0f}s", True, self.BLACK)
        self.screen.blit(time_label, (20, 460))

        if self.dispatch_mode == "rl" and hasattr(self.simulation.dispatch_center, 'dispatch_agent'):
            agent = self.simulation.dispatch_center.dispatch_agent
            model_info = f"ε: {agent.epsilon:.3f}"
            model_label = self.small_font.render("Model:", True, self.BLACK)
            self.screen.blit(model_label, (20, 490))
            model_display = self.small_font.render(model_info, True, self.BLUE)
            self.screen.blit(model_display, (20, 510))

    def _get_traffic_color_by_coord(self, road_coord):
        city_graph = self.simulation.city_graph

        if road_coord in city_graph.blocked_roads:
            return self.TRAFFIC_BLOCKED

        traffic_multiplier = city_graph.traffic_multipliers.get(road_coord, 1.0)

        if traffic_multiplier >= 2.0:
            return self.TRAFFIC_HEAVY
        elif traffic_multiplier >= 1.3:
            return self.TRAFFIC_MODERATE
        elif traffic_multiplier <= 0.9:
            return self.TRAFFIC_LIGHT
        else:
            return self.GRAY

    def draw_city(self):
        city_offset_x = self.control_panel_width

        pygame.draw.rect(self.screen, self.DARK_GRAY,
                        (city_offset_x, 0, self.simulation_width, self.height))

        for u, v, data in self.simulation.city_graph.graph.edges(data=True):
            node_u = self.simulation.city_graph.get_node_by_id(u)
            node_v = self.simulation.city_graph.get_node_by_id(v)

            if node_u and node_v:
                x1 = city_offset_x + int(node_u.x * self.scale_x)
                y1 = self.top_padding + int(node_u.y * self.scale_y)
                x2 = city_offset_x + int(node_v.x * self.scale_x)
                y2 = self.top_padding + int(node_v.y * self.scale_y)

                road_coord = (int(node_u.x), int(node_u.y))
                road_color = self._get_traffic_color_by_coord(road_coord)

                pygame.draw.line(self.screen, road_color, (x1, y1), (x2, y2), 2)

        pygame.draw.rect(self.screen, self.WHITE,
                        (city_offset_x + self.simulation_width, 0, self.sidebar_width, self.height))
        pygame.draw.line(self.screen, self.BLACK,
                        (city_offset_x + self.simulation_width, 0), (city_offset_x + self.simulation_width, self.height), 2)

    def draw_stations(self):
        city_offset_x = self.control_panel_width
        for station in self.simulation.stations:
            x = city_offset_x + int(station.location.x * self.scale_x)
            y = self.top_padding + int(station.location.y * self.scale_y)

            idle_vehicles = len(station.get_available_vehicles())

            pygame.draw.circle(self.screen, self.BLUE, (x, y), 18)
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 18, 2)

            station_text = self.small_font.render(f"R{station.id}", True, self.WHITE)
            station_rect = station_text.get_rect(center=(x, y - 5))
            self.screen.blit(station_text, station_rect)

            vehicle_text = self.small_font.render(f"({idle_vehicles})", True, self.WHITE)
            vehicle_rect = vehicle_text.get_rect(center=(x, y + 8))
            self.screen.blit(vehicle_text, vehicle_rect)

    def draw_hospitals(self):
        city_offset_x = self.control_panel_width
        for hospital in self.simulation.hospitals:
            x = city_offset_x + int(hospital.location.x * self.scale_x)
            y = self.top_padding + int(hospital.location.y * self.scale_y)
            pygame.draw.rect(self.screen, self.GREEN,
                           (x - 15, y - 10, 30, 20))

            text = self.small_font.render(f"H{hospital.id}", True, self.WHITE)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def draw_vehicles(self):
        city_offset_x = self.control_panel_width
        for vehicle in self.simulation.vehicles:
            if vehicle.status == VehicleStatus.IDLE:
                continue

            x = city_offset_x + int(vehicle.current_location.x * self.scale_x)
            y = self.top_padding + int(vehicle.current_location.y * self.scale_y)

            if vehicle.status == VehicleStatus.EN_ROUTE_TO_INCIDENT:
                color = self.YELLOW
            elif vehicle.status == VehicleStatus.AT_INCIDENT:
                color = self.ORANGE
            elif vehicle.status == VehicleStatus.TRANSPORTING_TO_HOSPITAL:
                color = self.PURPLE
            elif vehicle.status == VehicleStatus.RETURNING_TO_STATION:
                color = self.BLUE
            else:
                color = self.BLACK

            pygame.draw.circle(self.screen, color, (x, y), 12)
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 12, 2)

            text = self.small_font.render(str(vehicle.id), True, self.BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

            if len(vehicle.current_path) > 0:
                self.draw_vehicle_path(vehicle)

    def draw_vehicle_path(self, vehicle):
        city_offset_x = self.control_panel_width
        if len(vehicle.current_path) < 2:
            return

        points = []
        for node_id in vehicle.current_path[vehicle.path_index:]:
            node = self.simulation.city_graph.get_node_by_id(node_id)
            if node:
                x = city_offset_x + int(node.x * self.scale_x)
                y = self.top_padding + int(node.y * self.scale_y)
                points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.screen, self.RED, False, points, 2)

    def draw_incidents(self):
        city_offset_x = self.control_panel_width
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
            x = city_offset_x + int(incident_data["location_x"] * self.scale_x)
            y = self.top_padding + int(incident_data["location_y"] * self.scale_y)
            pygame.draw.circle(self.screen, self.RED, (x, y), 8)

    def draw_stats(self):
        metrics = self.simulation.log_data.get_performance_metrics()
        stats_text = [
            f"Simulation Time: {self.simulation.env.now:.0f} seconds",
            f"Total Incidents: {metrics.get('total_incidents', 0)}",
            f"Active Dispatches: {metrics.get('total_dispatches', 0)}",
            f"Avg Response Time: {metrics.get('avg_response_time', 0):.0f}s",
            f"Available Vehicles: {len(self.simulation.dispatch_center.get_available_vehicles())}/{len(self.simulation.vehicles)}",
        ]

        sidebar_x = self.control_panel_width + self.simulation_width + 20

        title = self.font.render("SIMULATION METRICS", True, self.BLACK)
        self.screen.blit(title, (sidebar_x, 20))

        for i, text in enumerate(stats_text):
            if len(text) > 25:
                text = text[:22] + "..."
            rendered = self.small_font.render(text, True, self.BLACK)
            self.screen.blit(rendered, (sidebar_x, 60 + i * 25))

        vehicle_status_counts = {}
        for vehicle in self.simulation.vehicles:
            status = vehicle.status.value
            vehicle_status_counts[status] = vehicle_status_counts.get(status, 0) + 1

        status_title = self.font.render("VEHICLE STATUS", True, self.BLACK)
        self.screen.blit(status_title, (sidebar_x, 220))

        y_offset = 250
        for status, count in vehicle_status_counts.items():
            status_text = f"{status.replace('_', ' ').title()}: {count}"
            rendered = self.small_font.render(status_text, True, self.BLACK)
            self.screen.blit(rendered, (sidebar_x, y_offset))
            y_offset += 20

    def draw_legend(self):
        legend_items = [
            ("Normal Roads", self.GRAY),
            ("Light Traffic", self.TRAFFIC_LIGHT),
            ("Moderate Traffic", self.TRAFFIC_MODERATE),
            ("Heavy Traffic", self.TRAFFIC_HEAVY),
            ("Road Blocked", self.TRAFFIC_BLOCKED),
            ("Buildings", self.DARK_GRAY),
            ("Response Stations", self.BLUE),
            ("Hospitals", self.GREEN),
            ("Active Incidents", self.RED),
            ("Vehicle Responding", self.YELLOW),
            ("Vehicle At Scene", self.ORANGE),
            ("Vehicle To Hospital", self.PURPLE)
        ]

        sidebar_x = self.control_panel_width + self.simulation_width + 20

        title = self.font.render("LEGEND", True, self.BLACK)
        self.screen.blit(title, (sidebar_x, 380))

        legend_x = sidebar_x
        legend_y = 420

        for i, (label, color) in enumerate(legend_items):
            y_pos = legend_y + i * 30
            pygame.draw.circle(self.screen, color, (legend_x + 8, y_pos + 10), 8)
            if color == self.WHITE:
                pygame.draw.circle(self.screen, self.BLACK, (legend_x + 8, y_pos + 10), 8, 2)
            text = self.small_font.render(label, True, self.BLACK)
            self.screen.blit(text, (legend_x + 25, y_pos + 2))

    def setup_simulation(self):
        self.env = self.simulation.env
        self.env.process(self.simulation.generate_incident())
        self.env.process(self.simulation.dispatch_process())
        self.env.process(self.simulation.vehicle_movement_process())
        self.env.process(self.simulation.incident_resolution_process())
        self.env.process(self.simulation.traffic_management_process())

        self.simulation.city_graph.add_traffic_incident(1800, "severe", 0)
        self.simulation.city_graph.add_traffic_incident(1200, "moderate", 0)
        self.simulation.city_graph.add_traffic_incident(900, "light", 0)

        self.simulation.city_graph.update_traffic_conditions(0)

    def step_simulation(self):
        if not self.simulation_completed and self.env.now < self.simulation.simulation_time:
            try:
                current_time = self.env.now
                self.env.step()
                if self.env.now == current_time:
                    self.env.run(until=current_time + 0.1)
                return True
            except Exception as e:
                print(f"Simulation error at time {self.env.now}: {e}")
                traceback.print_exc()
                self.simulation_completed = True
                self.simulation.log_summary_data()
                return False
        else:
            self.simulation_completed = True
            if not hasattr(self, '_logged_summary'):
                self.simulation.log_summary_data()
                self._logged_summary = True
            return False

    def handle_button_click(self, pos):
        if self.next_button.collidepoint(pos) and not self.simulation_completed:
            self.step_requested = True
        elif self.auto_button.collidepoint(pos) and not self.simulation_completed:
            self.auto_mode = not self.auto_mode
            self.simulation_paused = not self.auto_mode
        elif self.speedup_button.collidepoint(pos):
            current_index = self.speed_options.index(self.speed_multiplier)
            next_index = (current_index + 1) % len(self.speed_options)
            self.speed_multiplier = self.speed_options[next_index]
        elif self.mode_button.collidepoint(pos):
            self.dispatch_mode = "rl" if self.dispatch_mode == "heuristic" else "heuristic"
            self.reset_simulation()
        elif self.shift_button.collidepoint(pos):
            self.shift_mode = not self.shift_mode
            self.reset_simulation()
        elif self.train_button.collidepoint(pos) and self.dispatch_mode == "rl":
            if not self.training_active:
                self.start_training()
        elif self.reset_button.collidepoint(pos):
            self.reset_simulation()

    def reset_simulation(self):
        if self.shift_mode:
            self.simulation = Simulation(dispatch_mode=self.dispatch_mode, shift_mode=True)
        else:
            self.simulation = Simulation(simulation_time=300.0, random_seed=42,
                                       dispatch_mode=self.dispatch_mode, shift_mode=False)
        self.setup_simulation()
        self.auto_mode = False
        self.step_requested = False
        self.simulation_paused = True
        self.simulation_completed = False
        self.training_active = False
        if hasattr(self, '_logged_summary'):
            delattr(self, '_logged_summary')

    def start_training(self):
        self.training_active = True
        print(f"Starting RL training in background...")

        def training_thread():
            try:
                if self.shift_mode:
                    print("Running shift training (20 shifts)")
                    self.simulation.run_shift_training(num_shifts=20)
                else:
                    print("Running episode training (50 episodes)")
                    self.simulation.run_training_episodes(num_episodes=50, episode_length=300.0)
                print("Training completed!")
            except Exception as e:
                print(f"Training failed: {e}")
            finally:
                self.training_active = False

        training_thread = threading.Thread(target=training_thread, daemon=True)
        training_thread.start()

    def run(self):
        self.setup_simulation()
        clock = pygame.time.Clock()
        auto_step_counter = 0

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_button_click(event.pos)

            if self.auto_mode and not self.simulation_completed:
                auto_step_counter += 1
                steps_needed = max(1, 30 // self.speed_multiplier)
                if auto_step_counter >= steps_needed:
                    for _ in range(self.speed_multiplier):
                        if not self.step_simulation():
                            break
                    auto_step_counter = 0
            elif self.step_requested and not self.simulation_completed:
                self.step_simulation()
                self.step_requested = False

            self.screen.fill(self.WHITE)

            self.draw_control_panel()
            self.draw_city()
            self.draw_stations()
            self.draw_hospitals()
            self.draw_vehicles()
            self.draw_incidents()
            self.draw_stats()
            self.draw_legend()

            if self.simulation_completed:
                sidebar_x = self.control_panel_width + self.simulation_width + 20
                completion_text = self.font.render("SIMULATION COMPLETED", True, self.RED)
                self.screen.blit(completion_text, (sidebar_x, self.height - 50))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def main():
    simulation = Simulation(simulation_time=300.0, random_seed=42, dispatch_mode="heuristic")
    visualizer = SimulationVisualizer(simulation, "heuristic")
    visualizer.run()


if __name__ == "__main__":
    main()