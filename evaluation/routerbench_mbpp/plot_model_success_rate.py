import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "/Users/indersharma/Developer/router/evaluation/routerbench_mbpp"
configs = ["cost_efficient", "mixed", "skill_based"]

dfs = []
for config in configs:
    path = os.path.join(base_dir, config, "per_prompt.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # We need selected_model and selected_benchmark_performance
        dfs.append(df[['selected_model', 'selected_benchmark_performance', 'mode']])

# Combine all data
combined_df = pd.concat(dfs, ignore_index=True)

# Calculate success rate and count per model (overall)
model_stats = combined_df.groupby('selected_model').agg(
    success_rate=('selected_benchmark_performance', 'mean'),
    count=('selected_benchmark_performance', 'count')
).reset_index()

# Sort by count for better visualization, or by success rate
model_stats = model_stats.sort_values(by='count', ascending=False)

print("Model Success Rates (Overall):")
print(model_stats)

sns.set_theme(style="whitegrid")

# Plot overall success rate per model
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x='success_rate', 
    y='selected_model', 
    data=model_stats, 
    palette='viridis'
)
plt.title("Success Rate of Models Selected by the Router")
plt.xlabel("Success Rate (Performance = 1.0)")
plt.ylabel("Selected Model")
plt.xlim(0, 1.05)

# Add counts as text on the bars
for i, p in enumerate(ax.patches):
    ax.annotate(f"{model_stats.iloc[i]['success_rate']:.2f} (n={model_stats.iloc[i]['count']})", 
                (p.get_width() + 0.01, p.get_y() + p.get_height() / 2.), 
                va='center')

plt.tight_layout()
graphs_dir = os.path.join(base_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)
plt.savefig(os.path.join(graphs_dir, "model_success_rate.png"))
plt.close()

# Let's also do it per configuration (mode)
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='selected_benchmark_performance', 
    y='selected_model', 
    hue='mode', 
    data=combined_df, 
    errorbar=None, # Just show the mean success rate
    palette='Set2'
)
plt.title("Success Rate of Selected Models per Configuration")
plt.xlabel("Success Rate")
plt.ylabel("Selected Model")
plt.xlim(0, 1.05)
plt.legend(title='Configuration')
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "model_success_rate_per_mode.png"))
plt.close()

print(f"Graphs saved to {graphs_dir}")
