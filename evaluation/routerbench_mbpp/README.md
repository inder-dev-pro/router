# RouterBench MBPP Evaluation

This document outlines the evaluation results of the router across three different configurations on the `routerbench_mbpp` dataset:
1. **Cost Efficient**
2. **Mixed**
3. **Skill Based**

## Summary of Results

The table below summarizes the key metrics for each configuration based on 427 prompts.

| Metric | Cost Efficient | Mixed | Skill Based |
|---|---|---|---|
| **Success Rate** | 50.1% | 57.1% | 61.6% |
| **Average Cost** | $0.00018 | $0.00027 | $0.00399 |
| **Average Latency (ms)** | 1842.5 | 2111.7 | 2366.4 |
| **Average Regret** | 0.070 | 0.186 | 0.251 |
| **Oracle Agreement Rate** | 16.9% | 11.5% | 5.6% |
| **Zero Regret Percentage** | 16.9% | 11.5% | 74.9% |

## Proper Comparisons & Insights

### 1. Success Rate vs. Cost Trade-off
- **Cost Efficient**: Achieves a baseline success rate of ~50% with an extremely low average cost ($0.00018). This mode heavily prefers cheaper models, sacrificing some capability for significant cost savings.
- **Mixed**: Provides a balanced approach, raising the success rate to ~57% while keeping costs relatively low ($0.00027). This mode effectively balances utility and cost, routing to capable models only when necessary.
- **Skill Based**: Achieves the highest success rate at ~61.6% but incurs a significantly higher cost ($0.00399) compared to the other modes. It strongly prefers high-capability models (e.g., GPT-4) to maximize task success, regardless of the cost penalty.

### 2. Latency Considerations
- The **Cost Efficient** mode has the lowest latency (1842ms on average). Cheaper and smaller models generally infer faster.
- The **Skill Based** mode is the slowest (2366ms), which aligns with the heavy usage of larger, more complex models.
- **Mixed** mode sits exactly in the middle, indicating a proportional distribution of requests between fast/cheap models and slow/expensive ones.

### 3. Regret and Utility
- **Cost Efficient** has the lowest regret (0.070), primarily because in this configuration, "regret" heavily penalizes unnecessary spending. By consistently picking cheap models, it avoids cost-related regret.
- **Skill Based** has a higher average regret (0.251) because it often chooses expensive models even when a cheaper model could have sufficed. However, it boasts a very high zero regret percentage (74.9%), indicating that when it succeeds, it's often the only option that could have succeeded, or the utility of success outweighs the cost penalty in its specific objective function.
- **Oracle Agreement** drops significantly in Skill Based mode (5.6%) because the Oracle considers both cost and success, whereas Skill Based mode primarily optimizes for success, leading to disagreements on what the "optimal" route is.

## Generated Graphs
A comprehensive set of bar charts and a summary table comparing these metrics visually can be found in `summary_results.pdf`.
