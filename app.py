from flask import Flask, render_template, request, redirect
import joblib
import pandas as pd
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://tubaarpacay@localhost/steam_success_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


model = joblib.load("models/steam_success_model.pkl")
features = joblib.load("models/model_features.pkl")


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
    "vr",
]


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

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )


def get_form_data():
    data = {}

    for feature in features:
        value = request.form.get(feature)

        if value is None:
            data[feature] = 0
        elif value == "on":
            data[feature] = 1
        else:
            try:
                data[feature] = float(value)
            except ValueError:
                data[feature] = 0

    return data


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
        suggestions.append("To increase success probability, improve player engagement, store page quality, tag diversity and content depth.")

    if not suggestions:
        if prediction == 1:
            suggestions.append("The entered values look strong. Keep improving visibility, player retention and store presentation.")
        else:
            suggestions.append("To become more successful, the game should improve content depth, visibility, playtime and Steam page quality.")

    return suggestions


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    successful_rate = None
    unsuccessful_rate = None
    suggestions = []
    selected_tags = []
    selected_features = {}

    if request.method == "POST":
        data = get_form_data()

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

        if prediction == 1:
            prediction_text = "This game is predicted to be successful."
        else:
            prediction_text = "This game is predicted to be unsuccessful."

        selected_tags = get_selected_tags(data)

        selected_features = {
            "Price": data.get("price", 0),
            "Achievements": data.get("achievements", 0),
            "Average Playtime": data.get("average_playtime", 0),
            "Median Playtime": data.get("median_playtime", 0),
            "Required Age": data.get("required_age", 0),
            "English Support": "Yes" if data.get("english", 0) == 1 else "No",
            "Release Year": data.get("release_year", 0),
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
        prediction_text=prediction_text,
        successful_rate=successful_rate,
        unsuccessful_rate=unsuccessful_rate,
        suggestions=suggestions,
        selected_tags=selected_tags,
        selected_features=selected_features,
    )


@app.route("/history")
def history():
    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
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

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)