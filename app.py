from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import difflib

app = Flask(__name__)

movies_data = pd.read_csv('movies (1).csv')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(combined_features)


X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, np.zeros(X_train.shape[0]))  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        movie_features = combined_features.iloc[index_of_the_movie]
        movie_features_vectorized = vectorizer.transform([movie_features])
        predicted_class = model.predict(movie_features_vectorized)[0]
        recommended_movies = []
        for i in range(1, 21):  
            predicted_classes = model.predict(X_test)
            indices = np.where(predicted_classes == predicted_class)[0]
            for index in indices:
                title_from_index = movies_data[movies_data.index == index]['title'].values[0]
                recommended_movies.append(title_from_index)
        return render_template('index.html', recommended_movies=recommended_movies)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)