import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports", exist_ok=True)

steam_df = pd.read_csv("data/raw/steam.csv")
tag_df = pd.read_csv("data/raw/steamspy_tag_data.csv")

print("Steam Shape:")
print(steam_df.shape)

print("\nTag Shape:")
print(tag_df.shape)

merged_df = steam_df.merge(
    tag_df,
    on="appid",
    how="left"
)

print("\nMerged Shape:")
print(merged_df.shape)
print("\nMerged Columns Count:")
print(len(merged_df.columns))

merged_df.to_csv("data/processed/steam_merged.csv", index=False)

tag_columns = [
    col for col in tag_df.columns
    if col != "appid"
]

tag_summary = []

for tag in tag_columns:
    game_count = (merged_df[tag] > 0).sum()
    average_tag_value = merged_df[tag].mean()

    tag_summary.append({
        "tag": tag,
        "game_count": game_count,
        "average_tag_value": average_tag_value
    })

tag_summary_df = pd.DataFrame(tag_summary)

tag_summary_df = tag_summary_df.sort_values(
    by="game_count",
    ascending=False
)

tag_summary_df.to_csv(
    "reports/tag_summary.csv",
    index=False
)

print("\nTop 20 Most Common Tags:")
print(tag_summary_df.head(20))

print("\nMerged dataset saved successfully.")
print("Tag summary saved to reports/tag_summary.csv")