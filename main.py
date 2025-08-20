#Importing libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

#Importing the dataset 
dataset = pd.read_csv ("Serie_A.csv")

########    Features Engineering    ############ 
# Initialize lists for pre-match averages
team_avg_GF = []
team_avg_GA = []
team_avg_xG = []
team_avg_xGA = []
team_avg_Poss = []

opp_avg_GF = []
opp_avg_GA = []
opp_avg_xG = []
opp_avg_xGA = []
opp_avg_Poss = []

# Loop through each match
for idx, row in dataset.iterrows():
    season = row['Season']
    team = row['Team']
    opponent = row['Opponent']
    round_num = row['Round']
    
    # Previous matches for team in the same season
    team_history = dataset[(dataset['Season'] == season) &
                           (dataset['Team'] == team) &
                           (dataset['Round'] < round_num)]
    
    # Previous matches for opponent in the same season
    opp_history = dataset[(dataset['Season'] == season) &
                          (dataset['Team'] == opponent) &
                          (dataset['Round'] < round_num)]
    
    # Compute averages for team
    team_avg_GF.append(team_history['GF'].mean() if not team_history.empty else np.nan)
    team_avg_GA.append(team_history['GA'].mean() if not team_history.empty else np.nan)
    team_avg_xG.append(team_history['xG'].mean() if not team_history.empty else np.nan)
    team_avg_xGA.append(team_history['xGA'].mean() if not team_history.empty else np.nan)
    team_avg_Poss.append(team_history['Poss'].mean() if not team_history.empty else np.nan)
    
    # Compute averages for opponent
    opp_avg_GF.append(opp_history['GF'].mean() if not opp_history.empty else np.nan)
    opp_avg_GA.append(opp_history['GA'].mean() if not opp_history.empty else np.nan)
    opp_avg_xG.append(opp_history['xG'].mean() if not opp_history.empty else np.nan)
    opp_avg_xGA.append(opp_history['xGA'].mean() if not opp_history.empty else np.nan)
    opp_avg_Poss.append(opp_history['Poss'].mean() if not opp_history.empty else np.nan)

# Add pre-match average features to dataset
dataset['Team_GF_avg'] = team_avg_GF
dataset['Team_GA_avg'] = team_avg_GA
dataset['Team_xG_avg'] = team_avg_xG
dataset['Team_xGA_avg'] = team_avg_xGA
dataset['Team_Poss_avg'] = team_avg_Poss

dataset['Opponent_GF_avg'] = opp_avg_GF
dataset['Opponent_GA_avg'] = opp_avg_GA
dataset['Opponent_xG_avg'] = opp_avg_xG
dataset['Opponent_xGA_avg'] = opp_avg_xGA
dataset['Opponent_Poss_avg'] = opp_avg_Poss

# Keep only pre-match features + categorical info for prediction
pre_match_features = [
    'Season', 'Round', 'Team', 'Opponent', 'Venue',
    'Team_GF_avg', 'Team_GA_avg', 'Team_Poss_avg',
    'Opponent_GF_avg', 'Opponent_GA_avg', 'Opponent_Poss_avg',
    'Result'  # target
]

dataset_pre_match = dataset[pre_match_features].copy()  # Create a copy to avoid SettingWithCopyWarning

# Save the new dataset with features
dataset_pre_match.to_csv("Serie_A_features.csv", index=False)

print("New dataset with pre-match average features saved as 'Serie_A_features.csv'")

#Replacing the missing data 
imputer = SimpleImputer (strategy='mean')
X_numerical_features = ['Team_GF_avg','Team_GA_avg','Team_Poss_avg',
                       'Opponent_GF_avg','Opponent_GA_avg','Opponent_Poss_avg']
dataset_pre_match.loc[:,X_numerical_features] = imputer.fit_transform (dataset_pre_match[X_numerical_features])

#Label Encoder for the stadium 
stadium = ['Venue']
le = LabelEncoder ()
dataset_pre_match.loc[:, 'Venue'] = le.fit_transform (dataset_pre_match['Venue'])

#Extracting the features and the target 
X = dataset_pre_match.drop("Result", axis=1)
y = dataset_pre_match["Result"]

#Encoding the features
categorical_features = ['Season', 'Round', 'Team', 'Opponent']
ct = ColumnTransformer (transformers=[('cat',OneHotEncoder(drop='first', sparse_output=False),categorical_features)], remainder='passthrough') #OneHotEnocding with dense output
X = ct.fit_transform (X)

#Encoding the target column
le1 = LabelEncoder ()
y = le1.fit_transform (y)

#Splitting the dataset
X_train , X_test, y_train , y_test = train_test_split (X,y , test_size= 0.2, random_state=0)

#Creating the pipeline with StandardScaler that can handle sparse matrices
pipeline = Pipeline ([
    ('scaler', StandardScaler(with_mean=False)),  # Set with_mean=False to handle sparse matrices
    ('classifier', RandomForestClassifier(random_state=0))
    
    ])

# Parameters grid to tune the classifier 
params = [
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__kernel': ['rbf'],
        'classifier__gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'classifier__kernel': ['linear']
    },
    {
        'classifier': [XGBClassifier(eval_metric='logloss')]
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [5, 10, 15, 100],
        'classifier__weights': ['uniform', 'distance']
    },
    {
        'classifier': [BernoulliNB()],
        'classifier__fit_prior': [True, False],
        'classifier__alpha': [1.0, 0.5, 2.0]  
    }
]

# GridSearchCV with 10-fold CV to find best hyperparameters
grid = GridSearchCV(pipeline, param_grid=params, cv=10, scoring='accuracy', n_jobs=-1)

# Training the GridSearchCV
print("Training models with GridSearchCV...")
grid.fit(X_train, y_train)

print("Best model:", grid.best_estimator_)
print("Best hyperparameters:", grid.best_params_)
print("Best accuracy score:", grid.best_score_)

# Saving the best model
joblib.dump(grid.best_estimator_, "best_model.pkl")

# Predicting on test set using the best model
best_model = joblib.load("best_model.pkl")
y_pred = best_model.predict(X_test)

# Evaluation metrics on test set
print("\nTest Set Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))















