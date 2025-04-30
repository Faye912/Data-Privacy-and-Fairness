#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# load data
adult_data = fetch_openml(name='adult', version=2, as_frame=True)
df = adult_data.frame

# clean data
df = df.replace('?', np.nan).dropna()

# convert data types for consistency
df['age'] = df['age'].astype(int)
df['sex'] = df['sex'].astype(str)
df['class'] = df['class'].astype(str)

# define true coounts 
total_count = len(df)
query_results = {
    'over_50': (df['age'] > 50).sum() / total_count,
    'over_50k': (df['class'] == '>50K').sum() / total_count,
    'female': (df['sex'] == 'Female').sum() / total_count
}

#%%
# define Laplace mechanism
def laplace_mechanism(true_value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise

# apply Laplace mechanism for different epsilon values
epsilons = [0.01, 0.1, 0.5, 1.0]
results = []

for eps in epsilons:
    for query_name, true_val in query_results.items():
        noisy_val = laplace_mechanism(true_val, sensitivity=1/total_count, epsilon=eps)
        error = abs(noisy_val - true_val)
        results.append({
            'epsilon': eps,
            'query': query_name,
            'true_value': true_val,
            'noisy_value': noisy_val,
            'abs_error': error
        })

#%%
# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)
results_df


# %%
# visualize results 
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="epsilon", y="abs_error", hue="query", marker="o")
plt.title("Impact of Epsilon on Absolute Error in Laplace Mechanism")
plt.xlabel("Epsilon (Privacy Level)")
plt.ylabel("Absolute Error")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
