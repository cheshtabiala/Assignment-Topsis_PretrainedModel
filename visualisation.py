import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import result
data = pd.read_csv('result.csv')

# Display the table
print("Model Ranking Table:")
print(data[['Model', 'BERTScore', 'Training_time', 'METEOR', 'Rank']].sort_values(by='Rank'))

# Bar chart
labels = data['Model']
num_models = len(labels)

# Parameters for bar chart
bertscore_scores = data['BERTScore']
length_of_summary = data['Training_time']
meteor_scores = data['METEOR']
ranks = data['Rank']

# Normalize ranks to a scale of 0 to 1 for better comparison
normalized_ranks = ranks / np.max(ranks)

# Plot the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.2  # Reduced the bar width for better separation
index = np.arange(num_models)  # Use np.arange for evenly spaced bars

# Use different colors for each metric
ax.bar(index - bar_width, bertscore_scores, width=bar_width, label='BERTScore', color='purple')
ax.bar(index, length_of_summary, width=bar_width, label='Training_time', color='green', alpha=0.7)
ax.bar(index + bar_width, meteor_scores, width=bar_width, label='METEOR', color='blue', alpha=0.7)
ax.bar(index + 2 * bar_width, normalized_ranks, width=bar_width, label='Normalized Rank', color='red', alpha=0.7)

ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.set_ylabel('Metrics')
ax.set_title('Text Summarization Model Comparison')

ax.legend()
plt.savefig('barchart.png')
plt.show()
