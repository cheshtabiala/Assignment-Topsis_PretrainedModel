import pandas as pd
import numpy as np

# Load data from the CSV file
data = pd.read_csv('Inputfile.csv')

# Define relevant columns
model_col = 'Model'
bertscore_col = 'BERTScore'
length_col = 'Length_of_Summary'
meteor_col = 'METEOR'

# Extract relevant columns for selected models
selected_models = ['T5', 'BERT', 'XLNet', 'BART','Pegasus']
selected_data = data[data[model_col].isin(selected_models)]

# Extract metric values for selected models
bertscore_values = selected_data[bertscore_col].values
length_of_summary = selected_data[length_col].values
meteor_values = selected_data[meteor_col].values

# Weights for each parameter
weights = np.array([0.4, 0.2, 0.3])

# Normalize the matrix
normalized_matrix = np.column_stack([
    bertscore_values / np.max(bertscore_values),
    1 - (length_of_summary / np.max(length_of_summary)),
    meteor_values / np.max(meteor_values)
])

# Calculate the weighted normalized decision matrix
weighted_normalized_matrix = normalized_matrix * weights

# Ideal and Negative Ideal solutions
ideal_solution = np.max(weighted_normalized_matrix, axis=0)
negative_ideal_solution = np.min(weighted_normalized_matrix, axis=0)

# Calculate the separation measures
distance_to_ideal = np.sqrt(np.sum((weighted_normalized_matrix - ideal_solution)**2, axis=1))
distance_to_negative_ideal = np.sqrt(np.sum((weighted_normalized_matrix - negative_ideal_solution)**2, axis=1))

# Calculate the TOPSIS scores
topsis_scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

# Add TOPSIS scores to the selected data
selected_data['TOPSIS_Score'] = topsis_scores

# Rank the models based on TOPSIS scores
selected_data['Rank'] = selected_data['TOPSIS_Score'].rank(ascending=False)

# Print the results
print("Model Ranking:")
print(selected_data[[model_col, 'TOPSIS_Score', 'Rank']].sort_values(by='Rank'))

# Save the results to a CSV file
selected_data.to_csv('result.csv', index=False)
