from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
import numpy as np


load_dotenv()

app = Flask(__name__)

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError("DATABASE_URL is not defined. Please check your .env file.")

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


model = joblib.load("models/steam_success_model.pkl")
features = joblib.load("models/model_features.pkl")


tag_features = [
    "action", "adventure", "rpg", "indie", "strategy", "simulation", "casual",
    "sports", "racing", "multiplayer", "singleplayer", "horror", "survival",
    "zombies", "open_world", "platformer", "puzzle", "shooter", "anime",
    "e_sports", "fps", "co_op", "early_access", "free_to_play",
    "great_soundtrack", "atmospheric", "funny", "story_rich", "difficult",
    "sci_fi", "controller", "visual_novel", "retro", "pixel_graphics",
    "local_co_op", "local_multiplayer", "sandbox", "space", "war", "vr"
]


validation_rules = {
    "price": (0, 300, "Price must be between 0 and 300."),
    "achievements": (0, 1000, "Achievements must be between 0 and 1000."),
    "average_playtime": (0, 100000, "Average playtime must be between 0 and 100000."),
    "median_playtime": (0, 100000, "Median playtime must be between 0 and 100000."),
    "required_age": (0, 18, "Required age must be between 0 and 18."),
    "release_year": (1997, 2100, "Release year must be between 1997 and 2100."),
    "screenshot_count": (0, 100, "Screenshot count must be between 0 and 100."),
    "movie_count": (0, 50, "Movie count must be between 0 and 50."),
    "category_count": (0, 50, "Category count must be between 0 and 50."),
    "genre_count": (0, 30, "Genre count must be between 0 and 30."),
    "steamspy_tag_count": (0, 50, "Steam tag count must be between 0 and 50."),
    "short_description_length": (0, 500, "Short description length must be between 0 and 500."),
    "about_game_length": (0, 10000, "About game length must be between 0 and 10000."),
    "detailed_description_length": (0, 15000, "Detailed description length must be between 0 and 15000."),
    "minimum_req_length": (0, 5000, "Minimum requirements length must be between 0 and 5000."),
    "recommended_req_length": (0, 5000, "Recommended requirements length must be between 0 and 5000."),
}


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(30))
    success_rate = db.Column(db.Float)
    failure_rate = db.Column(db.Float)

    price = db.Column(db.Float)
    achievements = db.Column(db.Integer)
    average_playtime = db.Column(db.Float)
    median_playtime = db.Column(db.Float)
    required_age = db.Column(db.Integer)
    release_year = db.Column(db.Integer)

    selected_tags = db.Column(db.Text)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


with app.app_context():
    db.create_all()


def get_form_data():
    data = {}

    for feature in features:
        value = request.form.get(feature)

        if value is None:
            data[feature] = 0
        else:
            try:
                data[feature] = float(value)
            except ValueError:
                data[feature] = 0

    return data


def validate_input_data(data):
    errors = []

    for field, rule in validation_rules.items():
        min_value, max_value, message = rule
        value = data.get(field, 0)

        if value < min_value or value > max_value:
            errors.append(message)

    return errors


def get_selected_tags(data):
    selected_tags = []

    for tag in tag_features:
        if data.get(tag, 0) == 1:
            selected_tags.append(tag.replace("_", " ").title())

    return selected_tags


def get_suggestions(data, selected_tags, unsuccessful_rate, prediction):
    suggestions = []

    if data.get("price", 0) > 30:
        suggestions.append("The price should be more accessible for a wider audience.")

    if data.get("achievements", 0) < 20:
        suggestions.append("The game should include more achievements to increase player engagement.")

    if data.get("average_playtime", 0) < 300:
        suggestions.append("Average playtime should be increased with more content and replayability.")

    if data.get("median_playtime", 0) < 200:
        suggestions.append("Median playtime should be improved by strengthening the early-game experience.")

    if data.get("english", 0) == 0:
        suggestions.append("English support should be added to reach more international players.")

    if len(selected_tags) < 4:
        suggestions.append("More relevant Steam tags should be selected to improve discoverability.")

    if data.get("required_age", 0) >= 18:
        suggestions.append("The age restriction should be considered because it may reduce the potential audience.")

    if data.get("screenshot_count", 0) < 5:
        suggestions.append("The Steam page should include more screenshots to present the game better.")

    if data.get("movie_count", 0) < 1:
        suggestions.append("At least one gameplay video or trailer should be added.")

    if data.get("genre_count", 0) < 2:
        suggestions.append("The game should have clear and diverse genre information.")

    if data.get("steamspy_tag_count", 0) < 5:
        suggestions.append("The game should use more SteamSpy tags for better classification.")

    if data.get("short_description_length", 0) < 80:
        suggestions.append("The short description should be more informative and attractive.")

    if data.get("about_game_length", 0) < 500:
        suggestions.append("The about section should describe gameplay, features and value more clearly.")

    if unsuccessful_rate is not None and unsuccessful_rate >= 50:
        suggestions.append(
            "To increase success probability, improve player engagement, store page quality, tag diversity and content depth."
        )

    if not suggestions:
        if prediction == 1:
            suggestions.append(
                "The entered values look strong. Keep improving visibility, player retention and store presentation."
            )
        else:
            suggestions.append(
                "To become more successful, the game should improve content depth, visibility, playtime and Steam page quality."
            )

    return suggestions


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    prediction_label = None
    successful_rate = None
    unsuccessful_rate = None
    confidence_score = None
    confidence_level = None
    risk_level = None
    suggestions = []
    selected_tags = []
    selected_features = {}
    error_message = None

    if request.method == "POST":
        data = get_form_data()
        validation_errors = validate_input_data(data)

        if validation_errors:
            error_message = " ".join(validation_errors)

            return render_template(
                "index.html",
                active_page="prediction",
                prediction_text=prediction_text,
                prediction_label=prediction_label,
                successful_rate=successful_rate,
                unsuccessful_rate=unsuccessful_rate,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                risk_level=risk_level,
                suggestions=suggestions,
                selected_tags=selected_tags,
                selected_features=selected_features,
                error_message=error_message
            )

        input_df = pd.DataFrame([data])
        input_df = input_df[features]

        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            unsuccessful_rate = round(float(probabilities[0]) * 100, 2)
            successful_rate = round(float(probabilities[1]) * 100, 2)
        else:
            successful_rate = 100 if prediction == 1 else 0
            unsuccessful_rate = 0 if prediction == 1 else 100

        prediction_label = "Successful" if prediction == 1 else "Unsuccessful"

        prediction_text = (
            "This game is predicted to be successful."
            if prediction == 1
            else "This game is predicted to be unsuccessful."
        )

        confidence_score = max(successful_rate, unsuccessful_rate)

        if confidence_score >= 75:
            confidence_level = "High"
        elif confidence_score >= 55:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        if unsuccessful_rate >= 70:
            risk_level = "High Risk"
        elif unsuccessful_rate >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        selected_tags = get_selected_tags(data)

        selected_features = {
            "Price": data.get("price", 0),
            "Achievements": data.get("achievements", 0),
            "Average Playtime": data.get("average_playtime", 0),
            "Median Playtime": data.get("median_playtime", 0),
            "Required Age": data.get("required_age", 0),
            "Release Year": data.get("release_year", 0),
            "English Support": "Yes" if data.get("english", 0) == 1 else "No",
            "Windows Support": "Yes" if data.get("windows", 0) == 1 else "No",
            "Mac Support": "Yes" if data.get("mac", 0) == 1 else "No",
            "Linux Support": "Yes" if data.get("linux", 0) == 1 else "No",
            "Screenshot Count": data.get("screenshot_count", 0),
            "Movie Count": data.get("movie_count", 0),
            "Category Count": data.get("category_count", 0),
            "Genre Count": data.get("genre_count", 0),
            "Steam Tag Count": data.get("steamspy_tag_count", 0),
        }

        suggestions = get_suggestions(
            data=data,
            selected_tags=selected_tags,
            unsuccessful_rate=unsuccessful_rate,
            prediction=prediction
        )

        saved_prediction = Prediction(
            prediction=prediction_label,
            success_rate=float(successful_rate),
            failure_rate=float(unsuccessful_rate),
            price=float(data.get("price", 0)),
            achievements=int(data.get("achievements", 0)),
            average_playtime=float(data.get("average_playtime", 0)),
            median_playtime=float(data.get("median_playtime", 0)),
            required_age=int(data.get("required_age", 0)),
            release_year=int(data.get("release_year", 0)),
            selected_tags=", ".join(selected_tags)
        )

        db.session.add(saved_prediction)
        db.session.commit()

    return render_template(
        "index.html",
        active_page="prediction",
        prediction_text=prediction_text,
        prediction_label=prediction_label,
        successful_rate=successful_rate,
        unsuccessful_rate=unsuccessful_rate,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        risk_level=risk_level,
        suggestions=suggestions,
        selected_tags=selected_tags,
        selected_features=selected_features,
        error_message=error_message
    )


@app.route("/history")
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()

    total_predictions = len(predictions)
    successful_predictions = 0
    unsuccessful_predictions = 0
    average_success_rate = 0

    if total_predictions > 0:
        successful_predictions = sum(1 for p in predictions if p.prediction == "Successful")
        unsuccessful_predictions = sum(1 for p in predictions if p.prediction == "Unsuccessful")
        average_success_rate = round(
            sum(p.success_rate for p in predictions) / total_predictions,
            2
        )

    return render_template(
        "history.html",
        active_page="history",
        predictions=predictions,
        total_predictions=total_predictions,
        successful_predictions=successful_predictions,
        unsuccessful_predictions=unsuccessful_predictions,
        average_success_rate=average_success_rate
    )


@app.route("/delete/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)

    db.session.delete(prediction)
    db.session.commit()

    return redirect("/history")


@app.route("/clear-history", methods=["POST"])
def clear_history():
    Prediction.query.delete()
    db.session.commit()

    return redirect("/history")


@app.route("/feature-importance")
def feature_importance():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_names = [
            str(features[i]).replace("_", " ").title()
            for i in indices
        ]

        feature_values = [
            float(importances[i])
            for i in indices
        ]

    else:
        fallback_data = [
            ("Average Playtime", 0.18),
            ("Median Playtime", 0.15),
            ("Price", 0.10),
            ("Achievements", 0.09),
            ("Release Year", 0.08),
            ("Steam Tag Count", 0.07),
            ("Category Count", 0.06),
            ("Genre Count", 0.05),
            ("Screenshot Count", 0.04),
        ]

        feature_names = [item[0] for item in fallback_data]
        feature_values = [item[1] for item in fallback_data]

    max_value = max(feature_values) if feature_values else 1

    relative_values = [
        round((value / max_value) * 100, 2) if max_value > 0 else 0
        for value in feature_values
    ]

    return render_template(
        "feature_importance.html",
        active_page="feature",
        feature_names=feature_names,
        feature_values=feature_values,
        relative_values=relative_values
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        active_page="about"
    )


if __name__ == "__main__":
    app.run()
