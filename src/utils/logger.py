import csv
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class LogData:
    def __init__(self, log_file: str = "simulation_log.json"):
        self.log_file = log_file
        self.history = []
        self.performance_metrics = {}

    def log_event(self, event_type: str, data: Dict[str, Any], timestamp: float):
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data,
            "real_time": datetime.now().isoformat()
        }
        self.history.append(event)

    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.history:
            return {}

        incidents = [event for event in self.history if event["event_type"] == "incident_resolved"]
        dispatches = [event for event in self.history if event["event_type"] == "vehicle_dispatched"]

        if incidents:
            response_times = [event["data"].get("response_time", 0) for event in incidents]

            avg_response = sum(response_times) / len(response_times) if response_times else 0
            variance = sum((rt - avg_response) ** 2 for rt in response_times) / len(response_times) if response_times else 0
            std_dev = variance ** 0.5

            failed_incidents = sum(1 for rt in response_times if rt > 30.0)

            self.performance_metrics = {
                "total_incidents": len(incidents),
                "total_dispatches": len(dispatches),
                "avg_response_time": avg_response,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "std_response_time": std_dev,
                "failed_incidents_over_30s": failed_incidents,
                "failed_incident_rate": (failed_incidents / len(incidents)) if incidents else 0,
                "total_events": len(self.history)
            }

        return self.performance_metrics

    def export_to_csv(self, filepath: str):
        if not self.history:
            return

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "event_type", "real_time"]

            if self.history:
                data_keys = set()
                for event in self.history:
                    if "data" in event and isinstance(event["data"], dict):
                        data_keys.update(event["data"].keys())
                fieldnames.extend(sorted(data_keys))

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for event in self.history:
                row = {
                    "timestamp": event["timestamp"],
                    "event_type": event["event_type"],
                    "real_time": event["real_time"]
                }

                if "data" in event and isinstance(event["data"], dict):
                    row.update(event["data"])

                writer.writerow(row)

    def save_to_json(self, filepath: Optional[str] = None):
        output_file = filepath or self.log_file
        with open(output_file, 'w') as f:
            json.dump({
                "history": self.history,
                "performance_metrics": self.get_performance_metrics()
            }, f, indent=2)

    def clear_log(self):
        self.history = []
        self.performance_metrics = {}