import warnings
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, url_for, request, redirect, render_template, session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user 

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = "thisisasecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///users.db"
Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

input_movies = ["TERMINATOR 3", "BAD BOYS II", "CHRONICLES OF THE NARNIA", "S.W.A.T.", "35 UP", "JOHNY ENGLISH", "KUNG FU PANDA", "PULP FICTION", "HARRY POTTER", "BABY'S DAY OUT", "COBRA", "THE PRESTIGE", "HAPPT FEET", "DELHI BELLY", "CARS 2", "THE SMURFS", "THE AVENGERS", "ENGLISH VINGLISH", "TAKEN", "PARANORMAL ACTIVITY", "WOLVERINE", "LONE RANGER", "THE HOBBIT", "FINAL DESTINATION"]
movie_id = [6537, 6548, 41566, 6595, 26712, 6550, 59784, 296, 4896, 5096, 6800, 48780, 49274, 88069, 87876, 88356, 122912, 99636, 96861, 97701, 103772, 103384, 106489, 71252]

ratings_data = pd.read_csv("./training_data/ratings.csv")

user_ids = list(ratings_data.userId.unique())
user_to_encoded = {value : index for index, value in enumerate(user_ids)}
encoded_to_user = {index : value for index, value in enumerate(user_ids)}

movie_ids = list(ratings_data.movieId.unique())
movie_to_encoded = {value : index for index, value in enumerate(movie_ids)}
encoded_to_movie = {index : value for index, value in enumerate(movie_ids)}

ratings_data["user"] = ratings_data.userId.map(user_to_encoded)
ratings_data["movie"] = ratings_data.movieId.map(movie_to_encoded)
ratings_data.rating = ratings_data.rating.values.astype(np.float32)

def format_output(movies, ratings):
    count = 0
    m, g, r, output = [], [], [], {}
    for movie in movies.itertuples():
        m.append(movie.title)
        r.append(round(ratings[count], 1))
        g.append(movie.genres)
        count += 1

    return {"movies" : m, "ratings" : r, "genres" : g}

def get_recommendations(mids, ratings):
    model = load_model("./recommendation-model/collaborative-filtering-recommendation-system.h5")
    movie_df = pd.read_csv("./training_data/movies.csv")

    user_id = ratings_data.userId.sample(1).iloc[0]
    movies_watched_by_user = ratings_data[ratings_data.userId == user_id]
    movies_watched_by_user.drop("timestamp", axis = 1, inplace = True)
    for i in range(len(mids)):
        movies_watched_by_user = movies_watched_by_user.append({"userId": user_id, "movieId": mids[i], "rating": ratings[i], "user": user_to_encoded[user_id], "movie": movie_to_encoded[mids[i]]}, ignore_index = True)

    for column in list(movies_watched_by_user):
        if column != "rating":
            movies_watched_by_user[column] = movies_watched_by_user[column].astype(int)

    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie_to_encoded.keys())))
    movies_not_watched = [[movie_to_encoded.get(x)] for x in movies_not_watched]

    user_encoder = user_to_encoded.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))

    ratings = model.predict([user_movie_array[:, 1], user_movie_array[:, 0]]).flatten()

    top_ratings_indices = ratings.argsort()[-10:][::-1]
    rat_norm = ratings[top_ratings_indices]
    rating_denorm = rat_norm * (5.0 - 0.5) + 0.5
    recommended_movie_ids = [encoded_to_movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]

    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    recommendations = format_output(recommended_movies, rating_denorm)
    return recommendations

class Login(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(50))
    username = db.Column(db.String(15), unique=True)
    password = db.Column(db.String(80), nullable = False)

    def __repr__(self):
        return '<User %r>' % self.username

@login_manager.user_loader
def load_user(user_id):
    return Login.query.get(int(user_id))

class LoginForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message="Invalid Email"), Length(max = 50)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=4, max = 80)])

class SignupForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message="Invalid Email"), Length(max = 50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=4, max=80)])

@app.route('/', methods=['POST','GET'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Login.query.filter_by(email=form.email.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=True)
                return redirect(url_for('welcome'))
        return '<h1>Invalid Username or password</h1>'

    if request.method == "GET":
        return render_template('login.html', form = form)

@app.route('/signup', methods=['POST','GET'])
def signup():
    form = SignupForm()
    
    if form.validate_on_submit():
        secure_password = generate_password_hash(form.password.data, method='sha256')
        new_user = Login(email = form.email.data, username = form.username.data, password = secure_password)
        try: 
            db.session.add(new_user)
            db.session.commit()

            login_user(new_user, remember=True)
        except:
            pass
        return redirect(url_for('welcome'))

    if request.method == "GET":
        return render_template('signup.html', form = form)


@app.route("/welcome", methods = ["GET", "POST"])
@login_required
def welcome(input_field=""):
    if request.method == "POST":
        input_field = json.loads(request.get_data())
        mids = list(map(int, input_field["movie_ids"])) 
        ratings = list(map(float, input_field["ratings"]))

        output = get_recommendations(mids, ratings)
        session['movies'], session['ratings'], session['genres'] = json.dumps(output["movies"]), json.dumps(str(output["ratings"])), json.dumps(output["genres"]), 
        return "Got recommendations"
    return render_template("welcome.html", user = current_user.username, movies = input_movies, mid = movie_id)

@app.route("/logout", methods = ["GET"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("signup"))

@app.route("/result", methods = ["GET"])
def result():
    movies = json.loads(session["movies"])
    ratings = json.loads(session["ratings"])
    ratings = ratings[1:len(ratings) -1].replace(",", "")
    ratings = list(map(float, ratings.split())) 
    genres = json.loads(session["genres"])
    genres = [" | ".join(genre.split("|")) for genre in genres]
    return render_template("result.html", movies = movies, ratings = ratings, genres = genres)

if __name__ == "__main__":
    app.run(debug=True)

