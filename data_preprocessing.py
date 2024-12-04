import requests
from bs4 import BeautifulSoup
import pandas as pd

years = range(2020, 2025)  # Seasons from 2020 to 2024
all_data = []

for year in years:
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_totals.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'totals_stats'})
    df = pd.read_html(str(table))[0]
    df = df[df['Rk'] != 'Rk']
    df['Season'] = year  # Add a season column
    all_data.append(df)

# Combine all seasons into one DataFrame
final_df = pd.concat(all_data)
final_df.to_csv('nba_totals_2020_2024.csv', index=False)
print("Data for multiple seasons saved to 'nba_totals_2020_2024.csv'.")