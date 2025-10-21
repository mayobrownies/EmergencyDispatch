# AI-Driven Emergency Dispatch Optimization with Reinforcement Learning

A discrete-event simulation environment that trains and evaluates AI agents for optimizing emergency vehicle dispatch, demonstrating how Deep Q-Networks (DQN) compare to traditional heuristic approaches.

## Project Status

### What's Implemented
- **Complete Simulation Framework**: Discrete-event simulation using SimPy with city graph, vehicles, stations, and hospitals
- **DQN Agent**: Fully functional RL agent with experience replay, target networks, and epsilon-greedy exploration
- **Traffic Management**: Dynamic traffic conditions with random variations, and incident-based blockages
- **GUI Visualization**: Pygame interface showing vehicles, incidents, traffic conditions, and system metrics
- **Logging**: Performance metrics tracking and JSON export for analysis
- **Training**: Multi-episode training with shift-based and standard episode modes

### What's Still to Come
- **Tweaking RL Agent Parameters**: Try to improve RL performance through gradual parameter modifications 
- **Realistic Traffic Patterns**: More sophisticated traffic modeling based on real-world data
- **Bug Fix**: Traffic simulation issues beyond 20,000 seconds runtime
- **Improved Reward Function**: Improved reward signal incorporating multiple performance metrics
- **Larger City Grid**: Expanded simulation environment for more complex scenarios
- **Realistic Timings**: Realistic incident rates and response times
- **Performance Analysis**: Comprehensive comparison between heuristic and RL approaches

### Changes from Original Proposal
- **Added Traffic System**: Originally planned as future work, implemented early to demonstrate RL advantages (RL still needs improvements)
- **Enhanced GUI**: More sophisticated visualization than initially planned
- **Shift-based Training**: Added 8-hour shift simulation mode

## Installation Instructions

### Dependencies
- Python 3.8+
- See requirements.txt for package dependencies

### Setup
```bash
pip install -r requirements.txt
mkdir models
```

## Usage

### Starting the Application
```bash
# Launch the simulation GUI
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
- Colored circles: Active vehicles (yellow=responding, orange=at scene, purple=transporting, blue=returning)
- Colored roads: Traffic conditions (green=light, yellow=moderate, red=heavy, black=blocked)
- Red circles: Active incidents (orange=vehicle on scene)
- Blue circles: Stations with vehicle counts
- Green rectangles: Hospitals

### Expected Output/Behavior

**Heuristic Mode**:
- Vehicles dispatched to closest available incidents
- Real-time visualization with performance metrics
- Final statistics showing response times and utilization

**RL Mode**:
- Intelligent dispatch decisions based on trained model
- Adaptive behavior considering traffic and system state
- Current performance is worse than heuristic mode, but will hopefully improve with future modifications

**Training Mode**:
- Progress updates every 5 episodes
- Model checkpoints saved every 10 episodes
- Final model saved as `models/dqn_final.pth`

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
├── run_simulation.py             # Main GUI application and entry point
├── models/                       # Trained models
├── src/
    ├── simulation.py             # Core simulation orchestration
    ├── components/
    │   ├── vehicle.py            # Emergency vehicle implementation
    │   ├── incident.py           # Emergency incident modeling
    │   └── station.py            # Stations and hospitals
    ├── environment/
    │   ├── city_graph.py         # Road network with traffic system
    │   └── incident_generator.py # Stochastic incident generation
    ├── agents/
    │   ├── dispatch_agent.py     # DQN agent implementation
    │   └── dispatch_center.py    # Central dispatch coordination
    └── utils/
        └── logger.py             # Event logging and metrics
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

### Key Questions
1. How much improvement can RL provide over closest-vehicle heuristics?
2. Can RL agents learn to anticipate traffic patterns and future incidents?
3. What system features are most important for optimal dispatch decisions?
4. How does performance scale with city size and incident complexity?

## Technology Stack
- **Simulation Engine**: Python with SimPy for discrete-event simulation
- **Deep Learning**: PyTorch for DQN implementation
- **Graph Analysis**: NetworkX for city representation and pathfinding algorithms
- **Visualization**: Pygame for real-time GUI, Matplotlib and Seaborn for analysis
- **Data Processing**: NumPy and Pandas for numerical operations and data analysis