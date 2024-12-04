import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam

# Load the data
df = pd.read_csv('nba_2020_2024_merged_stats.csv')

# Calculate Fantasy Points
df['Fantasy_Points'] = (
    df['PTS'] +
    1.2 * df['TRB'] +
    1.5 * df['AST'] +
    3 * df['STL'] +
    3 * df['BLK'] -
    1 * df['TOV']
)

# Select relevant features
features = ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV%', 'FG%', 'FT%', 'PER', 'USG%', 'BPM',
            'TS%', 'ORtg', 'MP', 'PF']
X = df[features]
y = df['Fantasy_Points']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input data for LSTM (samples, time steps, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(1, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

# Make predictions for the 2024-25 season
season_2024_25 = df[df['Season'] == '2024-25'].copy()
X_2024_25 = season_2024_25[features]
X_2024_25_scaled = scaler.transform(X_2024_25)
X_2024_25_reshaped = X_2024_25_scaled.reshape((X_2024_25_scaled.shape[0], 1, X_2024_25_scaled.shape[1]))

predictions = model.predict(X_2024_25_reshaped)

# Add predictions to the DataFrame
season_2024_25['Predicted_Fantasy_Score'] = predictions

# Rank players by position
rankings = season_2024_25.groupby('Pos_x', group_keys=False).apply(lambda x: x.sort_values('Predicted_Fantasy_Score', ascending=False))

# Display top 10 players for each position
for position in rankings['Pos_x'].unique():
    print(f"\nTop 10 {position}s for 2024-25 Season:")
    print(rankings[rankings['Pos_x'] == position][['Player', 'Predicted_Fantasy_Score']].head(10))

# Write predictions to CSV
rankings[['Player', 'Pos_x', 'Season', 'Fantasy_Points', 'Predicted_Fantasy_Score']].to_csv('nba_fantasy_predictions_2024_25.csv', index=False)