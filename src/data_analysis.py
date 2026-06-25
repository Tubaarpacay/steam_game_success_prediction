import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("reports", exist_ok=True)

steam = pd.read_csv("data/raw/steam.csv")
tags = pd.read_csv("data/raw/steamspy_tag_data.csv")
description = pd.read_csv("data/raw/steam_description_data.csv")
media = pd.read_csv("data/raw/steam_media_data.csv")
requirements = pd.read_csv("data/raw/steam_requirements_data.csv")
support = pd.read_csv("data/raw/steam_support_info.csv")

print("\n========== DATASET OVERVIEW ==========")

datasets = {
    "steam.csv": steam,
    "steamspy_tag_data.csv": tags,
    "steam_description_data.csv": description,
    "steam_media_data.csv": media,
    "steam_requirements_data.csv": requirements,
    "steam_support_info.csv": support,
}

for name, df in datasets.items():
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print("Missing Values:")
    print(df.isnull().sum())


requirements = requirements.rename(columns={"steam_appid": "appid"})
support = support.rename(columns={"steam_appid": "appid"})
media = media.rename(columns={"steam_appid": "appid"})
description = description.rename(columns={"steam_appid": "appid"})

merged = steam.merge(tags, on="appid", how="left")
merged = merged.merge(requirements, on="appid", how="left")
merged = merged.merge(support, on="appid", how="left")
merged = merged.merge(media, on="appid", how="left")
merged = merged.merge(description, on="appid", how="left")

print("\n========== MERGED DATASET ==========")
print("Merged Shape:", merged.shape)
print("Merged Columns Count:", len(merged.columns))

merged.to_csv("data/processed/steam_merged.csv", index=False)
print("Merged dataset saved to data/processed/steam_merged.csv")


merged["total_ratings"] = (
    merged["positive_ratings"] + merged["negative_ratings"]
)

merged = merged[merged["total_ratings"] > 0]

merged["success_ratio"] = (
    merged["positive_ratings"] / merged["total_ratings"]
)

merged["successful"] = (
    merged["success_ratio"] >= 0.80
).astype(int)

print("\n========== SUCCESS ANALYSIS ==========")

print("\nSuccess Ratio Summary:")
print(merged["success_ratio"].describe())

print("\nSuccessful Games Distribution:")
print(merged["successful"].value_counts())

print("\nSuccessful Games Distribution Percentage:")
print(merged["successful"].value_counts(normalize=True) * 100)


plt.figure(figsize=(8, 5))
merged["successful"].value_counts().plot(kind="bar")
plt.title("Successful vs Unsuccessful Games")
plt.xlabel("Class")
plt.ylabel("Number of Games")
plt.tight_layout()
plt.savefig("reports/successful_distribution.png")
plt.close()


print("\n========== TOP GAMES ==========")

top_games = merged.sort_values(
    by="positive_ratings",
    ascending=False
)[["name", "positive_ratings", "negative_ratings", "success_ratio"]]

print(top_games.head(10))

top_games.head(20).to_csv(
    "reports/top_games_by_positive_ratings.csv",
    index=False
)


print("\n========== GENRE ANALYSIS ==========")

merged["main_genre"] = merged["genres"].fillna("").str.split(";").str[0]

genre_success = merged.groupby("main_genre")["success_ratio"].mean()

top_genres = genre_success.sort_values(ascending=False).head(10)

print("\nTop 10 Genres by Average Success Rate:")
print(top_genres)

top_genres.to_csv("reports/top_genres_by_success_rate.csv")

plt.figure(figsize=(10, 5))
top_genres.plot(kind="bar")
plt.title("Top 10 Genres by Average Success Rate")
plt.xlabel("Genre")
plt.ylabel("Average Success Rate")
plt.tight_layout()
plt.savefig("reports/top_genres_success_rate.png")
plt.close()


print("\n========== PLATFORM ANALYSIS ==========")

merged["windows"] = merged["platforms"].fillna("").str.contains("windows").astype(int)
merged["mac"] = merged["platforms"].fillna("").str.contains("mac").astype(int)
merged["linux"] = merged["platforms"].fillna("").str.contains("linux").astype(int)

platform_success = {
    "Windows": merged[merged["windows"] == 1]["success_ratio"].mean(),
    "Mac": merged[merged["mac"] == 1]["success_ratio"].mean(),
    "Linux": merged[merged["linux"] == 1]["success_ratio"].mean(),
}

platform_success_df = pd.DataFrame(
    list(platform_success.items()),
    columns=["Platform", "Average Success Ratio"]
)

print(platform_success_df)

platform_success_df.to_csv(
    "reports/platform_success_rate.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(
    platform_success_df["Platform"],
    platform_success_df["Average Success Ratio"]
)
plt.title("Average Success Rate by Platform Support")
plt.xlabel("Platform")
plt.ylabel("Average Success Ratio")
plt.tight_layout()
plt.savefig("reports/platform_success_rate.png")
plt.close()


print("\n========== TAG ANALYSIS ==========")

tag_columns = [
    col for col in tags.columns
    if col != "appid"
]

tag_summary = []

for tag in tag_columns:
    if tag in merged.columns:
        average_success = merged[merged[tag] > 0]["success_ratio"].mean()
        game_count = (merged[tag] > 0).sum()

        tag_summary.append({
            "tag": tag,
            "game_count": game_count,
            "average_success_ratio": average_success
        })

tag_summary_df = pd.DataFrame(tag_summary)

tag_summary_df = tag_summary_df.sort_values(
    by="average_success_ratio",
    ascending=False
)

print("\nTop 20 Tags by Average Success Ratio:")
print(tag_summary_df.head(20))

tag_summary_df.to_csv(
    "reports/tag_success_summary.csv",
    index=False
)

top_tags = tag_summary_df.head(10)

plt.figure(figsize=(10, 5))
plt.bar(
    top_tags["tag"],
    top_tags["average_success_ratio"]
)
plt.title("Top 10 Tags by Average Success Rate")
plt.xlabel("Tag")
plt.ylabel("Average Success Ratio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("reports/top_tags_success_rate.png")
plt.close()


print("\n========== NUMERIC CORRELATION ==========")

numeric_df = merged.select_dtypes(include=["int64", "float64"])

correlations = numeric_df.corr(numeric_only=True)

success_correlations = correlations["success_ratio"].sort_values(
    ascending=False
)

print("\nCorrelation with Success Ratio:")
print(success_correlations.head(30))

success_correlations.to_csv(
    "reports/success_ratio_correlations.csv"
)

plt.figure(figsize=(10, 8))
plt.imshow(correlations, aspect="auto")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()


print("\n========== PRICE ANALYSIS ==========")

price_analysis = merged.groupby("successful")["price"].describe()

print(price_analysis)

price_analysis.to_csv("reports/price_analysis.csv")


plt.figure(figsize=(8, 5))
merged.boxplot(column="price", by="successful")
plt.title("Price Distribution by Success Class")
plt.suptitle("")
plt.xlabel("Successful Class")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("reports/price_success_boxplot.png")
plt.close()


print("\n========== PLAYTIME ANALYSIS ==========")

playtime_analysis = merged.groupby("successful")[
    ["average_playtime", "median_playtime"]
].describe()

print(playtime_analysis)

playtime_analysis.to_csv("reports/playtime_analysis.csv")


print("\n========== FINAL SUMMARY ==========")

print("Raw steam rows:", steam.shape[0])
print("Merged rows:", merged.shape[0])
print("Merged columns:", len(merged.columns))
print("Reports saved inside reports/ folder.")
print("Processed dataset saved inside data/processed/ folder.")