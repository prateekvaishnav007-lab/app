import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory for EDA visuals
os.makedirs('outputs/eda_visuals', exist_ok=True)

# Load the cleaned data
df = pd.read_csv('data/protein_clean.csv')

# Summary statistics
print('Summary statistics:')
print(df.describe())

# Plot distributions for all features
features = df.columns.tolist()
plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(3, 4, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.savefig('outputs/eda_visuals/all_feature_distributions.png')
plt.close()

# Individual feature distributions
for col in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'outputs/eda_visuals/{col}_distribution.png')
    plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/eda_visuals/correlation_heatmap.png')
plt.close()

# Histogram of target variable (RMSD)
plt.figure(figsize=(6, 4))
sns.histplot(df['RMSD'], kde=True, bins=30)
plt.title('RMSD Distribution')
plt.xlabel('RMSD')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/eda_visuals/RMSD_distribution.png')
plt.close()