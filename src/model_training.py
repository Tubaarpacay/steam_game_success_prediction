import os
import ast
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def count_items(value):
    if pd.isna(value):
        return 0

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return len(parsed)
        if isinstance(parsed, dict):
            return len(parsed)
    except:
        return 0

    return 0


def text_length(value):
    if pd.isna(value):
        return 0

    return len(str(value))


def has_value(value):
    if pd.isna(value):
        return 0

    if str(value).strip() == "":
        return 0

    return 1


def owner_average(owner_range):
    if pd.isna(owner_range):
        return 0

    try:
        low, high = str(owner_range).split("-")
        low = int(low.replace(",", "").strip())
        high = int(high.replace(",", "").strip())
        return (low + high) / 2
    except:
        return 0


steam = pd.read_csv("data/raw/steam.csv")
tags = pd.read_csv("data/raw/steamspy_tag_data.csv")
requirements = pd.read_csv("data/raw/steam_requirements_data.csv")
support = pd.read_csv("data/raw/steam_support_info.csv")
media = pd.read_csv("data/raw/steam_media_data.csv")
description = pd.read_csv("data/raw/steam_description_data.csv")


requirements = requirements.rename(columns={"steam_appid": "appid"})
support = support.rename(columns={"steam_appid": "appid"})
media = media.rename(columns={"steam_appid": "appid"})
description = description.rename(columns={"steam_appid": "appid"})


df = steam.merge(tags, on="appid", how="left")
df = df.merge(requirements, on="appid", how="left")
df = df.merge(support, on="appid", how="left")
df = df.merge(media, on="appid", how="left")
df = df.merge(description, on="appid", how="left")


df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
df = df[df["total_ratings"] > 0].copy()

df["success_ratio"] = df["positive_ratings"] / df["total_ratings"]
df["successful"] = (df["success_ratio"] >= 0.80).astype(int)


engineered_features = pd.DataFrame(index=df.index)

engineered_features["release_year"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
).dt.year

engineered_features["release_year"] = engineered_features["release_year"].fillna(
    engineered_features["release_year"].median()
)

engineered_features["owners_average"] = df["owners"].apply(owner_average)

engineered_features["windows"] = df["platforms"].fillna("").str.contains(
    "windows",
    case=False
).astype(int)

engineered_features["mac"] = df["platforms"].fillna("").str.contains(
    "mac",
    case=False
).astype(int)

engineered_features["linux"] = df["platforms"].fillna("").str.contains(
    "linux",
    case=False
).astype(int)

engineered_features["category_count"] = df["categories"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(";"))
)

engineered_features["genre_count"] = df["genres"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(";"))
)

engineered_features["steamspy_tag_count"] = df["steamspy_tags"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(";"))
)

engineered_features["developer_count"] = df["developer"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(";"))
)

engineered_features["publisher_count"] = df["publisher"].fillna("").apply(
    lambda x: 0 if x == "" else len(str(x).split(";"))
)

engineered_features["has_website"] = df["website"].apply(has_value)
engineered_features["has_support_url"] = df["support_url"].apply(has_value)
engineered_features["has_support_email"] = df["support_email"].apply(has_value)

engineered_features["has_header_image"] = df["header_image"].apply(has_value)
engineered_features["has_background"] = df["background"].apply(has_value)

engineered_features["screenshot_count"] = df["screenshots"].apply(count_items)
engineered_features["movie_count"] = df["movies"].apply(count_items)

engineered_features["minimum_req_length"] = df["minimum"].apply(text_length)
engineered_features["recommended_req_length"] = df["recommended"].apply(text_length)

engineered_features["short_description_length"] = df["short_description"].apply(text_length)
engineered_features["about_game_length"] = df["about_the_game"].apply(text_length)
engineered_features["detailed_description_length"] = df["detailed_description"].apply(text_length)


basic_features = [
    "achievements",
    "average_playtime",
    "median_playtime",
    "price",
    "required_age",
    "english"
]

tag_features = [
    "action",
    "adventure",
    "rpg",
    "indie",
    "strategy",
    "simulation",
    "casual",
    "sports",
    "racing",
    "multiplayer",
    "singleplayer",
    "horror",
    "survival",
    "zombies",
    "open_world",
    "platformer",
    "puzzle",
    "shooter",
    "anime",
    "e_sports",
    "fps",
    "co_op",
    "early_access",
    "free_to_play",
    "great_soundtrack",
    "atmospheric",
    "funny",
    "story_rich",
    "difficult",
    "sci_fi",
    "controller",
    "visual_novel",
    "retro",
    "pixel_graphics",
    "local_co_op",
    "local_multiplayer",
    "sandbox",
    "space",
    "war",
    "vr"
]

engineered_feature_names = engineered_features.columns.tolist()

features = basic_features + engineered_feature_names + tag_features


for feature in basic_features + tag_features:
    if feature not in df.columns:
        df[feature] = 0


X = pd.concat(
    [
        df[basic_features + tag_features],
        engineered_features
    ],
    axis=1
)

X = X[features]
X = X.fillna(0)

y = df["successful"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nSuccessful Class Distribution:")
print(df["successful"].value_counts(normalize=True) * 100)


joblib.dump(model, "models/steam_success_model.pkl")
joblib.dump(features, "models/model_features.pkl")

processed_df = pd.concat(
    [
        df,
        engineered_features
    ],
    axis=1
)

processed_df.to_csv("data/processed/steam_merged.csv", index=False)


feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv("models/feature_importance.csv", index=False)

print("\nTop 20 Important Features:")
print(feature_importance.head(20))

print("\nModel saved successfully.")
print("Feature list saved successfully.")
print("Feature importance saved successfully.")
print("Merged dataset saved successfully.")