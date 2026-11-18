import pandas as pd

# Load the files (adjust filenames/paths if needed)
data_df = pd.read_csv('data.csv', sep='\t' if '\t' in open('data.csv').readline() else ',')
completed_path = 'corrected/gikuyu/lit_gikuyu_random_15.csv'
completed_df = pd.read_csv(completed_path)

# Option A: If matching by proverb text
if 'african_proverb' in data_df.columns and 'african_proverb' in completed_df.columns:
    predict_map = data_df.set_index('african_proverb')['predict'].to_dict()
    completed_df['predict'] = completed_df['african_proverb'].map(predict_map)

# Option B: If they align directly by row index order (uncomment below if needed)
# completed_df['predict'] = data_df['predict'].values[:len(completed_df)]

# Save back to the completed file
completed_df.to_csv(completed_path, index=False)
print(f"Successfully added 'predict' column to {completed_path}!")
