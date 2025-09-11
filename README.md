# An AI-Driven Simulation Environment for Optimizing Emergency Dispatch Policies

## Project Structure
```
├── .gitignore
├── README.md
├── config/
│   └── default_config.json
├── data/
│   └── city_graph.gml
├── notebooks/
│   └── 01_initial_analysis.ipynb
├── results/
│   ├── baseline/
│   └── rl_agent/
├── src/
│   ├── __init__.py
│   ├── simulation.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── vehicle.py
│   │   ├── incident.py
│   │   └── station.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── city_graph.py
│   │   └── incident_generator.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dispatch_agent.py
│   │   └── heuristic_agent.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
└── run_simulation.py
```

## Project Overview
This project develops a discrete-event simulation framework to explore optimal emergency dispatch policies using reinforcement learning.

### Domain
The simulation focuses on urban **Emergency Medical Services (EMS)** dispatch systems, a time-critical resource allocation challenge where rapid decision-making is crucial.

### Problem Statement
Most dispatch systems rely on a simple "closest available unit" heuristic, which can lead to suboptimal outcomes from a system-wide perspective. This project investigates:

> Can a reinforcement learning agent, trained in a realistic simulation environment, discover dispatch strategies that consistently outperform the standard closest-vehicle approach?

I hypothesize that an AI agent, given the ability to observe the entire system state and learn from thousands of simulated scenarios, will develop more sophisticated decision-making patterns.

### Planned Technologies
* **Simulation Engine**: Python with SimPy for discrete-event simulation
* **Reinforcement Learning**: PyTorch for DQN implementation
* **Spatial Analysis**: NetworkX for city graph representation and pathfinding
* **Data Analysis**: NumPy, Pandas, Matplotlib, Seaborn for results analysis

### Project Status
This repository contains the early file structure and project overview. Implementation will begin following Milestone 1 approval.