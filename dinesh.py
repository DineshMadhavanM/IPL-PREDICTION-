import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
players = ['Pant', 'Iyer', 'Kohli', 'Dhoni', 'Rohit', 'Rahul']
batting_avg = [50.5, 45.3, 52.7, 40.2, 50.0, 47.8]  
bowling_avg = [0, 35.0, 0, 0, 0, 0]  
age = [26, 28, 35, 39, 35, 30] 
past_ipl_runs = [3000, 2500, 6000, 5000, 4500, 3500] 
past_ipl_wickets = [0, 10, 0, 0, 0, 5]  
auction_price = (np.array(batting_avg) * 0.2) + (np.array(past_ipl_runs) * 0.01) + (np.array(past_ipl_wickets) * 0.05) + np.random.uniform(2, 5, size=6)
df = pd.DataFrame({
    'Player': players,
    'Batting Average': batting_avg,
    'Bowling Average': bowling_avg,
    'Age': age,
    'Past IPL Runs': past_ipl_runs,
    'Past IPL Wickets': past_ipl_wickets,
    'Auction Price': auction_price
})
print("Dataset Preview:")
print(df)
plt.figure(figsize=(10, 6))
sns.histplot(df['Auction Price'], kde=True, bins=30)
plt.title("Distribution of Auction Prices")
plt.xlabel("Auction Price (in Crores)")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap between Player Stats and Auction Price")
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Batting Average'], y=df['Auction Price'], color='blue')
plt.title("Auction Price vs Batting Average")
plt.xlabel("Batting Average")
plt.ylabel("Auction Price (in Crores)")
plt.show()
plt.figure(figsize=(10, 6))
sns.barplot(x='Age', y='Auction Price', data=df, palette="viridis")
plt.title("Auction Price vs Player Age")
plt.xlabel("Age")
plt.ylabel("Auction Price (in Crores)")
plt.show()
plt.figure(figsize=(12, 6))
sns.barplot(x='Player', y='Auction Price', data=df, palette='Blues_d')
plt.title("Auction Prices for Selected Players")
plt.xlabel("Player")
plt.ylabel("Auction Price (in Crores)")
plt.xticks(rotation=45, ha="right")
plt.show()
features = ['Batting Average', 'Bowling Average', 'Age', 'Past IPL Runs', 'Past IPL Wickets']
target = 'Auction Price'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
feature_importance = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importance)
plt.title("Feature Importance in Predicting Auction Price")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
new_player = np.array([[45.0, 30.0, 29, 4000, 15]]) 
predicted_price = model.predict(new_player)
print(f"Predicted Auction Price for the new player: {predicted_price[0]:.2f} Crores")
