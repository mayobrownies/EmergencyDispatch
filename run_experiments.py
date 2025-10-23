import os
import time
from src.simulation import Simulation

TRAINED_MODEL_SHIFT = r"models\dqn_shift_final.pth"
TRAINED_MODEL_EPISODE = r"models\dqn_episode_final.pth"

os.makedirs("results", exist_ok=True)

experiments = [
    {'id': 'run_01_heuristic_low_short',        'mode': 'heuristic', 'rate': 0.05,  'shift': False, 'time': 300},
    {'id': 'run_02_rl_low_short',               'mode': 'rl',        'rate': 0.05,  'shift': False, 'time': 300},
    {'id': 'run_03_heuristic_med_short',        'mode': 'heuristic', 'rate': 0.075, 'shift': False, 'time': 300},
    {'id': 'run_04_rl_med_short',               'mode': 'rl',        'rate': 0.075, 'shift': False, 'time': 300},
    {'id': 'run_05_heuristic_high_short',       'mode': 'heuristic', 'rate': 0.1,   'shift': False, 'time': 300},
    {'id': 'run_06_rl_high_short',              'mode': 'rl',        'rate': 0.1,   'shift': False, 'time': 300},
    {'id': 'run_07_heuristic_low_shift',        'mode': 'heuristic', 'rate': 0.002, 'shift': True},
    {'id': 'run_08_rl_low_shift',               'mode': 'rl',        'rate': 0.002, 'shift': True},
    {'id': 'run_09_heuristic_med_shift',        'mode': 'heuristic', 'rate': 0.005, 'shift': True},
    {'id': 'run_10_rl_med_shift',               'mode': 'rl',        'rate': 0.005, 'shift': True},
    {'id': 'run_11_heuristic_high_shift',       'mode': 'heuristic', 'rate': 0.008, 'shift': True},
    {'id': 'run_12_rl_high_shift',              'mode': 'rl',        'rate': 0.008, 'shift': True},
]

start_time = time.time()

for i, config in enumerate(experiments):
    print(f"\n--- Starting Experiment {i+1}/{len(experiments)}: {config['id']} ---")

    sim_time = (8 * 60 * 60) if config['shift'] else config.get('time', 300.0)

    if config['mode'] == 'rl':
        model_path = TRAINED_MODEL_SHIFT if config['shift'] else TRAINED_MODEL_EPISODE
        sim = Simulation(
            simulation_time=sim_time,
            dispatch_mode="rl",
            shift_mode=config['shift'],
            incident_rate=config['rate'],
            load_model_path=model_path,
            rl_training_mode=False
        )
    else:
        sim = Simulation(
            simulation_time=sim_time,
            dispatch_mode="heuristic",
            shift_mode=config['shift'],
            incident_rate=config['rate']
        )

    sim.run_simulation()

    output_filename = f"results/{config['id']}.json"
    sim.log_summary_data(output_filename)

    print(f"--- Completed Experiment. Results saved to {output_filename} ---")

print(f"\nAll 12 experiments completed in {(time.time() - start_time) / 60:.2f} minutes.")
