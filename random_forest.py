import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
predictions = pd.read_csv('nba_fantasy_predictions_2024_25.csv')
last_15_days = pd.read_csv('nba_last15days_2024-25.csv')

# Merge the datasets
merged_data = pd.merge(predictions, last_15_days, on='Player', how='inner')
print(merged_data.columns)

# Select features for the model
features = ['G', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG%', 'FT%', 'TS%', 'MP', 'BPM']
target = 'Predicted_Fantasy_Score'

# Prepare the data
X = merged_data[features]
y = merged_data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Predict for all players
all_predictions = rf_model.predict(X)
merged_data['RF_Predicted_Fantasy_Score'] = all_predictions

# Display top 10 players based on Random Forest predictions
top_10 = merged_data.nlargest(10, 'RF_Predicted_Fantasy_Score')
print("\nTop 10 Players based on Random Forest Predictions:")
print(top_10[['Player', 'Pos', 'Team_x', 'RF_Predicted_Fantasy_Score', 'G']])

# Save results to CSV
merged_data[['Player', 'Pos', 'Team_x', 'Predicted_Fantasy_Score', 'RF_Predicted_Fantasy_Score', 'G']].to_csv('rf_fantasy_predictions.csv', index=False)
print("\nResults saved to rf_fantasy_predictions.csv")