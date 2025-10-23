# AI-Driven Emergency Dispatch Optimization with Reinforcement Learning

A discrete-event simulation environment that trains and evaluates AI agents for optimizing emergency vehicle dispatch, demonstrating how Deep Q-Networks (DQN) compare to traditional heuristic approaches.

## Project Status

### What's Implemented
- **Complete Simulation Framework**: Discrete-event simulation using SimPy with real Atlanta street network via OSMnx
- **DQN Agent**: Fully functional RL agent with experience replay, target networks, and epsilon-greedy exploration
- **Traffic Management**: Rush-hour patterns, dynamic congestion, traffic incidents with severity levels and decay
- **GUI Visualization**: Pygame GUI showing vehicles, incidents, traffic conditions, and some real-time system metrics
- **Metrics and Logging**: Performance metrics including avg/max/std response times, failure rates, and JSON/CSV export
- **Training System**: Multi-episode training with 8-hour shift mode and episode mode (200 episodes each)
- **Configuration Management**: Config file (config.py) for main hyperparameters and simulation settings
- **Experiments**: Batch experiment runner with multiple load levels and dispatch modes (RL and heuristic)

## Installation Instructions

### Dependencies
- Python 3.8+
- See requirements.txt for package dependencies

### Setup
```bash
pip install -r requirements.txt
mkdir models results
```

## Usage

### Running Experiments
```bash
# Run all 12 experiments (heuristic + RL across different loads)
python run_experiments.py

# Analyze results
python analysis.py
```

### Starting the GUI
```bash
# Launch the GUI
python run_simulation.py
```

### GUI Interface
The application provides an interactive GUI where users can:

**Dispatch Mode Selection**
- Toggle between "Heuristic" and "RL" dispatch strategies using the on-screen button
- Heuristic mode uses closest-available-vehicle logic
- RL mode loads the best available trained model

**Training Controls**
- Click "Train RL" button to start training new models
- Training runs multiple episodes with progress updates
- Models are automatically saved during training

**Simulation Controls**
- Start/Stop simulation using GUI buttons
- Real-time visualization shows vehicles, incidents, and traffic conditions
- Performance metrics displayed on screen

**Visual Elements**
- Colored circles: Active vehicles
- Colored roads: Traffic conditions
- Red circles: Active incidents
- Blue circles: Stations with vehicle counts
- Green rectangles: Hospitals

### Expected Output/Behavior

**Heuristic Mode**:
- Vehicles dispatched to closest available incidents using traffic-aware pathfinding
- Typical average response time: ~15-17 seconds
- Low failure rate, but less consistent

**RL Mode**:
- Dispatch decisions based on trained DQN model
- Typical average response time: ~18-20 seconds
- More consistent performance (lower std deviation)
- Higher failure rate for edge cases (>30s incidents)

**Training Mode**:
- Progress updates every 5 episodes
- Model checkpoints saved every 10 episodes
- Final models saved to `models/` folder

**Experiment Output**:
- 12 JSON files in `results/` directory with complete metrics
- CSV summary table with all performance metrics
- 7 charts showing comparative performance

## Architecture Overview

### Main Components

**Simulation Core**
- `Simulation`: Orchestrates discrete-event simulation using SimPy
- `CityGraph`: NetworkX-based road network with traffic management
- `IncidentGenerator`: Poisson process for stochastic incident creation

**Entities**
- `Vehicle`: Emergency vehicles with status tracking and pathfinding
- `Incident`: Emergency events requiring response
- `Station/Hospital`: Infrastructure facilities

**AI Agent**
- `DispatchAgent`: DQN implementation with experience replay
- `DispatchCenter`: Central coordination hub using either heuristic or RL dispatch

**Visualization & Logging**
- GUI (run_simulation.py): Real-time pygame visualization
- `LogData`: Comprehensive event logging and performance metrics

### Mapping to UML Design

**Core Classes**: All major classes (Simulation, CityGraph, Vehicle, DispatchAgent) implemented as designed
**Traffic System**: Added to CityGraph as an extension of the original road network concept

### Architectural Changes and Rationale

**Traffic Integration**: Integrated throughout the system rather than as a separate module for better performance and realistic vehicle movement

**GUI Separation**: Visualization logic kept separate from core simulation for modularity and testing

**State Space Enhancement**: Expanded RL state representation to include traffic conditions, allowing the agent to eventually learn traffic-aware dispatch strategies

## Project Structure
```
├── README.md
├── requirements.txt
├── config.py
├── run_simulation.py
├── run_experiments.py
├── analysis.py
│
├── results/                      # Experiment outputs and visualizations
│   ├── run_*.json
│   ├── results.csv
│   └── chart_*.png
│
└── src/                          # Source code
    ├── simulation.py
    │
    ├── components/               # Simulation entities
    │   ├── vehicle.py
    │   ├── incident.py
    │   └── station.py
    │
    ├── environment/              # City and incident generation
    │   ├── city_graph.py
    │   └── incident_generator.py
    │
    ├── agents/                   # AI dispatch agents and coordination
    │   ├── dispatch_agent.py
    │   └── dispatch_center.py
    │
    └── utils/                    # Logging and metrics
        └── logger.py
```

## Domain & Problem Statement

### Emergency Medical Services (EMS) Dispatch
The simulation focuses on urban Emergency Medical Services dispatch systems, a time-critical resource allocation challenge where rapid decision-making is crucial.

### The Challenge
Traditional dispatch systems rely on simple heuristics like "send the closest available vehicle." While intuitive, this approach can lead to:
- Suboptimal system-wide resource distribution
- Increased overall response times
- Poor handling of multiple simultaneous incidents
- Inability to adapt to traffic patterns and city dynamics

## Technology Stack
- **Simulation Engine**: Python with SimPy for discrete-event simulation
- **Deep Learning**: PyTorch for DQN implementation
- **Graph**: NetworkX for city representation and pathfinding algorithms, OSMnx for real OpenStreetMap data integration
- **Visualization**: Pygame for real-time GUI, Matplotlib and Seaborn for analysis
- **Data Processing**: NumPy and Pandas for numerical operations and data analysis