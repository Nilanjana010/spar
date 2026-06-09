import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/path/to/global_metrics.csv')

metrics = ['edges', 'logit_diff', 'faithfulness', 'tsr', 'tsd']

eval_means = df.groupby('exponent')[metrics].mean().round(3)

display_names = {'edges': 'Edges', 'logit_diff': 'Logit Diff',
                 'faithfulness': 'Faithfulness', 'tsr': 'TSR', 'tsd': 'TSD'}
eval_means.columns = [display_names.get(c, c) for c in eval_means.columns]

#Plot
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table = ax.table(cellText=eval_means.values, colLabels=eval_means.columns,
                 rowLabels=[f"Exp {i}" for i in eval_means.index],
                 loc='center', cellLoc='center')

#Styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    elif col == -1:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#ecf0f1')

plt.title("Performance by Sparsity Exponent", pad=25, fontsize=14, weight='bold')
plt.savefig('summaryTable.png', bbox_inches='tight', dpi=300)
