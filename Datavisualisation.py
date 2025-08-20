# Importing the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Importing the new dataset with only pre-match features
dataset = pd.read_csv("Serie_A_features.csv")

# Shape of the dataset
print(f"Shape of the dataset (rows, columns): {dataset.shape}")

# Statistical description
print("Statistical description of the dataset: ")
print(dataset.describe())

# Class distribution (target variable)
print("\nClass distribution (target variable):")
print(dataset['Result'].value_counts())
print(dataset['Result'].value_counts(normalize=True) * 100)

# Plot histograms for all numerical pre-match features
numeric_features = [
    'Team_GF_avg', 'Team_GA_avg', 'Team_Poss_avg',
    'Opponent_GF_avg', 'Opponent_GA_avg', 'Opponent_Poss_avg'
]

dataset[numeric_features].hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots to visualize relationship between numerical features and the result
for feature in numeric_features:
    sns.boxplot(x='Result', y=feature, data=dataset)
    plt.xlabel(feature)
    plt.ylabel('Result')
    plt.title(f'Relation between {feature} and match result')
    plt.show()



