import pandas as pd

games_df = pd.read_csv("data/raw/steam.csv")

print("Dataset Shape:")
print(games_df.shape)

print("\nColumn Names:")
print(games_df.columns.tolist())

print("\nFirst Five Rows:")
print(games_df.head())

print("\nDataset Information:")
print(games_df.info())

print("\nMissing Values:")
print(games_df.isnull().sum())

games_df["total_ratings"] = games_df["positive_ratings"] + games_df["negative_ratings"]

games_df["success_ratio"] = games_df["positive_ratings"] / games_df["total_ratings"]

print("\nSuccess Ratio Summary:")
print(games_df["success_ratio"].describe())

games_df["successful"] = (games_df["success_ratio"] >= 0.80).astype(int)

print("\nSuccessful Games Distribution:")
print(games_df["successful"].value_counts())

import matplotlib.pyplot as plt

games_df["successful"].value_counts().plot(kind="bar")

plt.title("Successful vs Unsuccessful Games")
plt.xlabel("Class")
plt.ylabel("Number of Games")

plt.show()

print("\nTop 10 Most Popular Games:")

top_games = games_df.sort_values(
    by="positive_ratings",
    ascending=False
)[["name", "positive_ratings"]]

print(top_games.head(10))

games_df["main_genre"] = games_df["genres"].str.split(";").str[0]

genre_success = games_df.groupby("main_genre")["success_ratio"].mean()

print("\nAverage Success Rate by Genre:")
print(genre_success.sort_values(ascending=False).head(10))

top_genres = genre_success.sort_values(
    ascending=False
).head(10)

top_genres.plot(kind="bar")

plt.title("Top 10 Genres by Success Rate")
plt.xlabel("Genre")
plt.ylabel("Average Success Rate")

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

features = [
    "achievements",
    "average_playtime",
    "price",
    "required_age",
    "english"
]

X = games_df[features]

y = games_df["successful"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nDataset Columns:")
print(games_df.columns.tolist())

print("\nCorrelation with Success Ratio:")

correlations = games_df.select_dtypes(include=["int64", "float64"]).corr()

print(correlations["success_ratio"].sort_values(ascending=False))

import seaborn as sns

numeric_df = games_df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=False
)

plt.title("Correlation Heatmap")
plt.show()

print("\nSteam Description Columns:")
description_df = pd.read_csv("data/raw/steam_description_data.csv")
print(description_df.columns.tolist())

print("\nSteam Media Columns:")
media_df = pd.read_csv("data/raw/steam_media_data.csv")
print(media_df.columns.tolist())

print("\nSteam Requirements Columns:")
requirements_df = pd.read_csv("data/raw/steam_requirements_data.csv")
print(requirements_df.columns.tolist())

print("\nSteam Support Columns:")
support_df = pd.read_csv("data/raw/steam_support_info.csv")
print(support_df.columns.tolist())

print("\nSteamSpy Tag Columns:")
tag_df = pd.read_csv("data/raw/steamspy_tag_data.csv")
print("Number of tag columns:")
print(len(tag_df.columns))

print("\nFirst 20 tag columns:")
print(tag_df.columns[:20].tolist())

print("\nTag Data Shape:")
print(tag_df.shape)