import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

base_dir = "/Users/indersharma/Developer/router/evaluation/routerbench_mbpp"
configs = ["cost_efficient", "mixed", "skill_based"]
summaries = {}

for config in configs:
    path = os.path.join(base_dir, config, "summary.json")
    if os.path.exists(path):
        summaries[config] = load_summary(path)

# Prepare dataframe
data = []
for config, summary in summaries.items():
    data.append({
        "Configuration": config.replace("_", " ").title(),
        "Success Rate": summary.get("routed_task_success_rate", 0),
        "Avg Cost ($)": summary.get("avg_actual_cost", 0),
        "Avg Latency (ms)": summary.get("avg_router_latency_ms", 0),
        "Avg Regret": summary.get("avg_regret", 0),
        "Oracle Agreement Rate": summary.get("oracle_agreement_rate", 0)
    })

df = pd.DataFrame(data)

# Set up the plots directory
graphs_dir = os.path.join(base_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

from matplotlib.backends.backend_pdf import PdfPages
pdf_path = os.path.join(base_dir, "summary_results.pdf")

with PdfPages(pdf_path) as pdf:
    # 1. Summary Table Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    # Formatting for display
    display_df = df.copy()
    display_df['Success Rate'] = display_df['Success Rate'].apply(lambda x: f"{x:.3f}")
    display_df['Avg Cost ($)'] = display_df['Avg Cost ($)'].apply(lambda x: f"{x:.5f}")
    display_df['Avg Latency (ms)'] = display_df['Avg Latency (ms)'].apply(lambda x: f"{x:.1f}")
    display_df['Avg Regret'] = display_df['Avg Regret'].apply(lambda x: f"{x:.3f}")
    display_df['Oracle Agreement Rate'] = display_df['Oracle Agreement Rate'].apply(lambda x: f"{x:.3f}")
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("Summary Results Table", pad=20, fontsize=14, fontweight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # 2. Success Rate Plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Configuration", y="Success Rate", data=df, hue="Configuration", palette="viridis", legend=False)
    plt.title("Routed Task Success Rate by Configuration")
    plt.ylim(0, 1)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 3. Avg Cost Plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Configuration", y="Avg Cost ($)", data=df, hue="Configuration", palette="magma", legend=False)
    plt.title("Average Actual Cost by Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%.5f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 4. Avg Latency Plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Configuration", y="Avg Latency (ms)", data=df, hue="Configuration", palette="crest", legend=False)
    plt.title("Average Router Latency by Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 5. Avg Regret Plot
    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Configuration", y="Avg Regret", data=df, hue="Configuration", palette="rocket", legend=False)
    plt.title("Average Regret by Configuration")
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 6. Cost vs Success Rate Scatter (Trade-off)
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Avg Cost ($)", y="Success Rate", hue="Configuration", s=200, palette="deep")
    plt.title("Cost vs Success Rate Trade-off")
    for i in range(len(df)):
        plt.text(df["Avg Cost ($)"][i], df["Success Rate"][i] + 0.01, df["Configuration"][i], horizontalalignment='center')
    pdf.savefig(fig)
    plt.close()
    
    # 7. Model Success Rates (Overall)
    dfs = []
    for config in configs:
        csv_path = os.path.join(base_dir, config, "per_prompt.csv")
        if os.path.exists(csv_path):
            prompt_df = pd.read_csv(csv_path)
            dfs.append(prompt_df[['selected_model', 'selected_benchmark_performance', 'mode']])
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    model_stats = combined_df.groupby('selected_model').agg(
        success_rate=('selected_benchmark_performance', 'mean'),
        count=('selected_benchmark_performance', 'count')
    ).reset_index().sort_values(by='count', ascending=False)
    
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='success_rate', y='selected_model', data=model_stats, palette='viridis')
    plt.title("Overall Success Rate of Models Selected by the Router")
    plt.xlabel("Success Rate (Performance = 1.0)")
    plt.ylabel("Selected Model")
    plt.xlim(0, 1.05)
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{model_stats.iloc[i]['success_rate']:.2f} (n={model_stats.iloc[i]['count']})", 
                    (p.get_width() + 0.01, p.get_y() + p.get_height() / 2.), 
                    va='center')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # 8. Model Success Rates (Per Mode)
    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='selected_benchmark_performance', 
        y='selected_model', 
        hue='mode', 
        data=combined_df, 
        errorbar=None, 
        palette='Set2'
    )
    plt.title("Success Rate of Selected Models per Configuration")
    plt.xlabel("Success Rate")
    plt.ylabel("Selected Model")
    plt.xlim(0, 1.05)
    plt.legend(title='Configuration')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("Graphs and summary generated successfully in", pdf_path)
