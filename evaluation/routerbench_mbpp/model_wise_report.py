import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

new_run_csv = "/Users/indersharma/Developer/router/data/mbpp_benchmark_20260920_145226.csv"
raw_data_csv = "/Users/indersharma/Developer/router/data/routerbench_raw_mbpp.csv"
base_dir = "/Users/indersharma/Developer/router/evaluation/routerbench_mbpp"

# 1. Process New Run Data
df_new = pd.read_csv(new_run_csv)
df_raw = pd.read_csv(raw_data_csv)

# Merge to get the true benchmark performance for each routed decision
df_new = df_new.merge(df_raw[['sample_id', 'model_name', 'performance']], 
                      left_on=['sample_id', 'selected_model'], 
                      right_on=['sample_id', 'model_name'], 
                      how='left')

# Calculate overall model success rates across all modes in the new run
model_stats = df_new.groupby('selected_model').agg(
    success_rate=('performance', 'mean'),
    count=('performance', 'count')
).reset_index().sort_values(by='count', ascending=False)

print("--- Model-Wise Success Rates (New Run) ---")
print(model_stats.to_string(index=False))

pdf_path = os.path.join(base_dir, "model_wise_report.pdf")
sns.set_theme(style="whitegrid")

with PdfPages(pdf_path) as pdf:
    # 1. Overall Model Success Rate
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='success_rate', y='selected_model', data=model_stats, hue='selected_model', palette='viridis', legend=False)
    plt.title("Success Rate of Models Selected by Router (New Run - 100 Prompts)")
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
    
    # 2. Model Success Rates Per Mode
    df_new['routing_mode'] = df_new['routing_mode'].str.replace('_', ' ').str.title()
    fig = plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='performance', 
        y='selected_model', 
        hue='routing_mode', 
        data=df_new, 
        errorbar=None, 
        palette='Set2'
    )
    plt.title("Model Success Rate Per Configuration (New Run)")
    plt.xlabel("Success Rate")
    plt.ylabel("Selected Model")
    plt.xlim(0, 1.05)
    plt.legend(title='Configuration')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"\nModel-wise report generated successfully in {pdf_path}")
