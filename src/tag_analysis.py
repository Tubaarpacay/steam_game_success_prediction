
import pandas as pd

steam_df = pd.read_csv("data/raw/steam.csv")
tag_df = pd.read_csv("data/raw/steamspy_tag_data.csv")

print("Steam Shape:")
print(steam_df.shape)

print("\nTag Shape:")
print(tag_df.shape)

merged_df = steam_df.merge(
    tag_df,
    on="appid"
)

print("\nMerged Shape:")
print(merged_df.shape)

print("\nMerged Columns:")
print(len(merged_df.columns))

merged_df.to_csv("data/processed/steam_merged.csv", index=False)

print("\nMerged dataset saved successfully.")