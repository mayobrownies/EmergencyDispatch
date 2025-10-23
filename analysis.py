import pandas as pd
import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

results_path = "results/*.json"
all_results = []

for f in glob.glob(results_path):
    with open(f, 'r') as file:
        data = json.load(file)
        metrics = data.get('performance_metrics', {})
        filename = os.path.basename(f).split('.')[0]
        parts = filename.split('_')
        metrics['id'] = filename
        if len(parts) >= 5:
            metrics['dispatch_mode'] = parts[2]
            metrics['load'] = parts[3]
            metrics['duration'] = parts[4]
        all_results.append(metrics)

df = pd.DataFrame(all_results)
df['avg_response_time'] = pd.to_numeric(df['avg_response_time'], errors='coerce')
df['total_incidents'] = pd.to_numeric(df['total_incidents'], errors='coerce')
df['max_response_time'] = pd.to_numeric(df['max_response_time'], errors='coerce')
df['std_response_time'] = pd.to_numeric(df['std_response_time'], errors='coerce')
df['failed_incidents_over_30s'] = pd.to_numeric(df['failed_incidents_over_30s'], errors='coerce')
df['failed_incident_rate'] = pd.to_numeric(df['failed_incident_rate'], errors='coerce')

print(f"Loaded {len(df)} experiment results")
print(df.head())

print("\n--- Summary Table ---")
summary_table = df[['id', 'dispatch_mode', 'load', 'duration', 'avg_response_time', 'max_response_time',
                     'std_response_time', 'failed_incidents_over_30s', 'total_incidents']]

output_csv_file = "results/results.csv"
summary_table.to_csv(output_csv_file, index=False)

print("\n" + str(summary_table))

plt.figure(figsize=(10, 6))
short_runs = df[df['duration'] == 'short']
if not short_runs.empty:
    sns.barplot(data=short_runs, x='dispatch_mode', y='avg_response_time', hue='load')
    plt.title('Average Response Time (300-Second Demos)')
    plt.ylabel('Average Response Time (seconds)')
    plt.xlabel('Dispatch Mode')
    plt.savefig('results/chart_1_response_time_short.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGenerated: results/chart_1_response_time_short.png")
else:
    print("\nNo short duration experiments found")

plt.figure(figsize=(10, 6))
shift_runs = df[df['duration'] == 'shift']
if not shift_runs.empty:
    sns.barplot(data=shift_runs, x='dispatch_mode', y='avg_response_time', hue='load', hue_order=['low', 'med', 'high'])
    plt.title('Average Response Time (8-Hour Shifts)')
    plt.ylabel('Average Response Time (seconds)')
    plt.xlabel('Dispatch Mode')
    plt.savefig('results/chart_2_response_time_shift.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: results/chart_2_response_time_shift.png")
else:
    print("No shift experiments found")

plt.figure(figsize=(10, 6))
if not shift_runs.empty:
    sns.barplot(data=shift_runs, x='dispatch_mode', y='total_incidents', hue='load', hue_order=['low', 'med', 'high'])
    plt.title('Total Incidents Resolved (8-Hour Shifts)')
    plt.ylabel('Total Incidents')
    plt.xlabel('Dispatch Mode')
    plt.savefig('results/chart_3_incidents_resolved.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: results/chart_3_incidents_resolved.png")

print("\n=== Statistical Summary by Dispatch Mode ===")
summary_stats = df.groupby('dispatch_mode').agg({
    'avg_response_time': ['mean', 'std', 'min', 'max'],
    'total_incidents': ['mean', 'std', 'min', 'max']
})
print(summary_stats)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='load', y='avg_response_time', hue='dispatch_mode')
plt.title('Response Time Distribution by Load Level')
plt.ylabel('Average Response Time (seconds)')
plt.xlabel('Incident Load Level')
plt.ylim(14, 22)
plt.savefig('results/chart_4_response_time_by_load.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated: results/chart_4_response_time_by_load.png")

plt.figure(figsize=(10, 6))
if not shift_runs.empty:
    sns.barplot(data=shift_runs, x='dispatch_mode', y='max_response_time', hue='load', hue_order=['low', 'med', 'high'])
    plt.title('Maximum Response Time (8-Hour Shifts)')
    plt.ylabel('Max Response Time (seconds)')
    plt.xlabel('Dispatch Mode')
    plt.savefig('results/chart_5_max_response_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: results/chart_5_max_response_time.png")

plt.figure(figsize=(10, 6))
if not shift_runs.empty:
    sns.barplot(data=shift_runs, x='dispatch_mode', y='std_response_time', hue='load', hue_order=['low', 'med', 'high'])
    plt.title('Response Time Consistency (Std Deviation)')
    plt.ylabel('Standard Deviation (seconds)')
    plt.xlabel('Dispatch Mode')
    plt.savefig('results/chart_6_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: results/chart_6_consistency.png")

plt.figure(figsize=(10, 6))
if not shift_runs.empty:
    sns.barplot(data=shift_runs, x='dispatch_mode', y='failed_incident_rate', hue='load', hue_order=['low', 'med', 'high'])
    plt.title('Failed Incident Rate (>30s Response)')
    plt.ylabel('Failed Incident Rate (%)')
    plt.xlabel('Dispatch Mode')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}'))
    plt.savefig('results/chart_7_failure_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: results/chart_7_failure_rate.png")