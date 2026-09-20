import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

base_dir = "/Users/indersharma/Developer/router/evaluation/routerbench_mbpp"
configs = ["cost_efficient", "mixed", "skill_based"]

dfs = []
for config in configs:
    csv_path = os.path.join(base_dir, config, "per_prompt.csv")
    if os.path.exists(csv_path):
        prompt_df = pd.read_csv(csv_path)
        dfs.append(prompt_df[['selected_model', 'selected_benchmark_performance', 'mode']])

combined_df = pd.concat(dfs, ignore_index=True)

pdf_path = os.path.join(base_dir, "success_rates_report.pdf")
sns.set_theme(style="whitegrid")

with PdfPages(pdf_path) as pdf:
    # 1. Router Success Rate (Across Modes)
    # The success rate of the router is just the mean of selected_benchmark_performance for each mode
    router_stats = combined_df.groupby('mode')['selected_benchmark_performance'].mean().reset_index()
    router_stats.rename(columns={'selected_benchmark_performance': 'Success Rate', 'mode': 'Configuration'}, inplace=True)
    
    fig = plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Configuration", y="Success Rate", data=router_stats, hue="Configuration", palette="viridis", legend=False)
    plt.title("Overall Router Success Rate by Configuration")
    plt.ylim(0, 1)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
    pdf.savefig(fig)
    plt.close()

    # 2. Model Success Rates (Overall when routed)
    model_stats = combined_df.groupby('selected_model').agg(
        success_rate=('selected_benchmark_performance', 'mean'),
        count=('selected_benchmark_performance', 'count')
    ).reset_index().sort_values(by='count', ascending=False)
    
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='success_rate', y='selected_model', data=model_stats, hue='selected_model', palette='magma', legend=False)
    plt.title("Success Rate of Models When Selected by the Router (Overall)")
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

    # 3. Model Success Rates (Per Mode when routed)
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

print(f"Success rates report generated successfully in {pdf_path}")
