import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to clean and transform the data
def process_csv(file_path):
    # Load the data from a CSV file
    df = pd.read_csv(file_path)

    # Sum the 'medium' and 'large' columns to create a new column 'medium_large'
    df['medium_large'] = df['medium'] + df['large']

    # Drop the columns that are no longer needed
    df.drop(['medium', 'large', 'Unnamed: 0'], axis=1, inplace=True)

    # Optionally, if 'small' needs to be rounded to one decimal place
    df['small'] = df['small'].round(1)

    # Round the 'medium_large' column to nearest 0.4
    df['medium_large'] = (np.round(df['medium_large'] / 0.4) * 0.4).round(1)

    # Return the cleaned DataFrame
    return df

original_df = pd.read_csv('/scratch/jh7956/cleaned_data.csv')
# print the rows with the top 5 'f1_test' values
print(original_df.nlargest(5, 'f1_test'))

# Assuming the path to the file and processing it
df = process_csv('/scratch/jh7956/cleaned_data.csv')

# Assuming 'f1_test' is a column in the DataFrame after processing
# Group by 'small' and 'medium_large', then mean 'f1_test' if there are duplicates
df_grouped = df.groupby(['small', 'medium_large']).f1_test.mean().reset_index()

# Pivot the DataFrame to set 'small' as rows and 'medium_large' as columns for heatmap
heatmap_data = df_grouped.pivot(index='small', columns='medium_large', values='f1_test')

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(heatmap_data, cmap='Blues', annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Heatmap of Averaged F1 Test Scores')
plt.xlabel('Medium + Long Percentage')
plt.ylabel('Short Percentage')

# Save the plot
plt.savefig('/scratch/jh7956/heatmap.png')

plt.show()
