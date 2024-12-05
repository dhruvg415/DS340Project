import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam

#Load training data
train_df = pd.read_csv('nba_2020_2024_merged_stats.csv')
last_15_days = pd.read_csv('nba_last15days_2024-25.csv')

# Load prediction data
current_totals = pd.read_csv('nba_2024-25_totals.csv')
current_advanced = pd.read_csv('nba_2024-25_adv.csv')

# Merge totals and advanced stats for 2024-25 season
prediction_df = pd.merge(current_totals, current_advanced, on=['Player', 'Season', 'Team', 'Pos'], suffixes=('_totals', '_adv'))

# Calculate fantasy points for training data
def calculate_fantasy_points(df):
    df['Fantasy_Points'] = (
        df['PTS'] +
        1.2 * df['TRB'] +
        1.5 * df['AST'] +
        3 * df['STL'] +
        3 * df['BLK'] -
        1 * df['TOV']
    )
    return df

train_df = calculate_fantasy_points(train_df)

# Create rolling averages for last 15 days
last_15_days = last_15_days.sort_values(['Player', 'GS'])
rolling_features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG%', 'FT%']
for feature in rolling_features:
    last_15_days[f'{feature}_rolling_5'] = last_15_days.groupby('Player')[feature].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)

# Merge last 15 days data with current season data
prediction_df = pd.merge(prediction_df, last_15_days.groupby('Player').last().reset_index(), on='Player', suffixes=('', '_last15'))
print(prediction_df.columns)

# Select relevant features for training
features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'FG%', 'FT%', 'PER', 'USG%', 'BPM', 'MP']

X_train = train_df[features]
y_train = train_df['Fantasy_Points']

# Normalize training features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Reshape input data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))

# Define LSTM model
model = Sequential([
    LSTM(128, input_shape=(1, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# Ensure consistent feature set with training data
X_pred = prediction_df[features]

# Normalize prediction features using the same scaler
X_pred_scaled = scaler.transform(X_pred)

X_pred_rolling = prediction_df[rolling_features]
X_pred_combined = np.hstack((X_pred_scaled, X_pred_rolling))

# Reshape prediction data for LSTM
X_pred_reshaped = X_pred_combined.reshape((X_pred_combined.shape[0], 1, X_pred_combined.shape[1]))

# Make predictions
predictions = model.predict(X_pred_reshaped)
prediction_df['Predicted_Fantasy_Score'] = predictions

rankings = prediction_df.groupby('Pos').apply(lambda x: x.sort_values('Predicted_Fantasy_Score', ascending=False))

for position in rankings.index.get_level_values(0).unique():
    print(f"\nTop 10 {position}s for 2024-25 Season:")
    print(rankings.loc[position][['Player', 'Predicted_Fantasy_Score']].head(10))

# Save predictions to a CSV file
prediction_df[['Player', 'Team', 'Pos', 'Predicted_Fantasy_Score']].to_csv('nba_fantasy_predictions_2024_25.csv', index=False)

print("Predictions saved to nba_fantasy_predictions_2024_25.csv")