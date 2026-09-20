import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

new_run_csv = "/Users/indersharma/Developer/router/data/mbpp_benchmark_20260920_145226.csv"
raw_data_csv = "/Users/indersharma/Developer/router/data/routerbench_raw_mbpp.csv"
old_base_dir = "/Users/indersharma/Developer/router/evaluation/routerbench_mbpp"

# 1. Process New Run
df_new = pd.read_csv(new_run_csv)
df_raw = pd.read_csv(raw_data_csv)

# Merge to get performance and actual cost
df_new = df_new.merge(df_raw[['sample_id', 'model_name', 'performance', 'cost']], 
                      left_on=['sample_id', 'selected_model'], 
                      right_on=['sample_id', 'model_name'], 
                      how='left')

# Calculate stats for new run
new_stats = df_new.groupby('routing_mode').agg(
    success_rate=('performance', 'mean'),
    avg_cost=('cost', 'mean'),
    avg_latency=('latency_s', 'mean')
).reset_index()
new_stats['Run'] = 'New Run (Fixes Applied)'

# 2. Process Old Run
modes = ["cost_efficient", "mixed", "skill_based"]
old_dfs = []
for mode in modes:
    path = os.path.join(old_base_dir, mode, "per_prompt.csv")
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        # Filter to the same sample_ids as the new run for a fair comparison (limit 100)
        old_df = old_df[old_df['sample_id'].isin(df_new['sample_id'])]
        old_dfs.append(old_df[['mode', 'selected_benchmark_performance', 'selected_benchmark_cost', 'router_latency_ms']])

df_old = pd.concat(old_dfs, ignore_index=True)
old_stats = df_old.groupby('mode').agg(
    success_rate=('selected_benchmark_performance', 'mean'),
    avg_cost=('selected_benchmark_cost', 'mean'),
    avg_latency=('router_latency_ms', lambda x: x.mean() / 1000.0) # convert ms to s
).reset_index()
old_stats.rename(columns={'mode': 'routing_mode'}, inplace=True)
old_stats['Run'] = 'Old Run'

# Combine stats
combined_stats = pd.concat([old_stats, new_stats], ignore_index=True)
combined_stats['routing_mode'] = combined_stats['routing_mode'].str.replace('_', ' ').str.title()

pdf_path = os.path.join(old_base_dir, "comparison_report.pdf")
sns.set_theme(style="whitegrid")

with PdfPages(pdf_path) as pdf:
    # 1. Success Rate Comparison
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="routing_mode", y="success_rate", hue="Run", data=combined_stats, palette="Set1")
    plt.title("Success Rate Comparison: Old vs New Run (100 Prompts)")
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.xlabel("Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 2. Average Cost Comparison
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="routing_mode", y="avg_cost", hue="Run", data=combined_stats, palette="Set2")
    plt.title("Average Cost Comparison: Old vs New Run (100 Prompts)")
    plt.ylabel("Average Cost ($)")
    plt.xlabel("Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%.5f', padding=3)
    pdf.savefig(fig)
    plt.close()
    
    # 3. Model Distribution Comparison (New Run)
    # Count how often each model was selected in the new run
    model_counts_new = df_new.groupby(['routing_mode', 'selected_model']).size().reset_index(name='count')
    model_counts_new['routing_mode'] = model_counts_new['routing_mode'].str.replace('_', ' ').str.title()
    
    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="routing_mode", y="count", hue="selected_model", data=model_counts_new, palette="tab10")
    plt.title("Model Selection Distribution: New Run (100 Prompts)")
    plt.ylabel("Number of Times Selected")
    plt.xlabel("Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%d', padding=3)
    plt.legend(title='Selected Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
print(f"Comparison report generated successfully in {pdf_path}")
