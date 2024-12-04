import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the advanced stats CSV file
advanced_data = pd.read_csv('nba_2020_2024_advanced.csv')
print(advanced_data.head)

# Assuming you also have a totals data CSV loaded as 'totals_data'
totals_data = pd.read_csv('nba_2020_2024_totals.csv')
print(totals_data.head)

# Merge the datasets on 'Player' and 'Season'
combined_data = pd.merge(totals_data, advanced_data, on=['Player', 'Season'], suffixes=('_totals', '_adv'))
'''
# Select relevant features for modeling
features = ['Age', 'G', 'MP', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV',
            'PER', 'TS%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%',
            'TOV%', 'USG%', 'OWS', 'DWS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
categorical_features = ['Pos_totals', 'Team_totals']

# Calculate fantasy points (adjust formula as needed)
combined_data['Fantasy_Points'] = (
    combined_data['PTS'] +
    1.2 * combined_data['TRB'] +
    1.5 * combined_data['AST'] +
    3 * combined_data['STL'] +
    3 * combined_data['BLK'] -
    1 * combined_data['TOV']
)

# Prepare features and target variable
X = combined_data[features + categorical_features]
y = combined_data['Fantasy_Points']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions
train_predictions = model_pipeline.predict(X_train)
test_predictions = model_pipeline.predict(X_test)

# Evaluate the model
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"Train MAE: {train_mae}")
print(f"Test MAE: {test_mae}")

# Plot feature importance
feature_importance = model_pipeline.named_steps['regressor'].feature_importances_
feature_names = numeric_features + list(model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df.sort_values(by='importance', ascending=False, inplace=True)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# Generate fantasy rankings by position
positions = ['PG', 'SG', 'SF', 'PF', 'C']

for pos in positions:
    pos_players = combined_data[combined_data['Pos_totals'] == pos].copy()
    pos_players['Predicted_Fantasy_Points'] = model_pipeline.predict(pos_players[features + categorical_features])
    top_players = pos_players.nlargest(10, 'Predicted_Fantasy_Points')
    print(f"\nTop 10 {pos} players by Predicted Fantasy Points:")
    print(top_players[['Player', 'Predicted_Fantasy_Points']])
'''