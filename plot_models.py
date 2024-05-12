import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


csv = 'training_results.csv'

sns.set_style("whitegrid")

data = pd.read_csv(csv)

data['val_loss'] = pd.to_numeric(data['val_loss'].str.replace('tensor ', ''), errors='coerce')

filtered_data = pd.DataFrame()
for model_name in data['model_name'].unique():
    model_data = data[data['model_name'] == model_name]
    model_data = model_data.sort_values(by='trainable_params')
    to_keep = []
    for index, row in model_data.iterrows():
        if not any((model_data['trainable_params'] < row['trainable_params']) & (model_data['val_loss'] <= row['val_loss'])):
            to_keep.append(index)
    filtered_data = pd.concat([filtered_data, model_data.loc[to_keep]])

colors = {
    'mlp': 'red',
    'efficient-kan': 'blue',
    'FastKAN': 'green',
    'ChebyKAN': 'purple',
    'JacobiKAN': 'orange',
    'RBFKAN': 'brown'
}

plt.figure(figsize=(12, 8))
x_limits = (25000, 50000)
y_limits = (0.16, 0.275)
for model_name, group_data in filtered_data.groupby('model_name'):
    color = colors[model_name]
    scatter = sns.scatterplot(data=group_data, x='trainable_params', y='val_loss', color=color, label=model_name, s=150, alpha=0.7)

    # Annotate each point with [l1, l2], only if within the plot limits
    for line in range(0, group_data.shape[0]):
        x = group_data['trainable_params'].iloc[line]
        y = group_data['val_loss'].iloc[line]
        if x_limits[0] <= x <= x_limits[1] and y_limits[0] <= y <= y_limits[1]:
            plt.text(x + 0.1, y,
                     f"[{group_data['n_units_l1'].iloc[line]},{group_data['n_units_l2'].iloc[line]}]",
                     horizontalalignment='left',
                     color=color,
                     weight='semibold')

    # LOWESS trend line
    lowess = sm.nonparametric.lowess(group_data['val_loss'], group_data['trainable_params'], frac=0.6)
    plt.plot(lowess[:, 0], lowess[:, 1], color=color)

plt.xlabel('Number of Trainable Parameters', fontsize=14)
plt.ylabel('Validation Loss', fontsize=14)
plt.xscale('log')
plt.xlim(x_limits)
plt.ylim(y_limits)
plt.legend(title='Model Type', fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig('optimized_model_performance.png', format='png', dpi=300)
plt.show()
