from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load preprocessed data
movies = pickle.load(open('C:\\Users\\pc\\OneDrive\\Desktop\\movie recomndation system\\movie recommendation\\list_of_movies.pkl', 'rb'))
similarity = pickle.load(open('C:\\Users\\pc\\OneDrive\\Desktop\\movie recomndation system\\movie recommendation\\similarities.pkl', 'rb'))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html', movies=movies['title'].values)

# Recommendation API
@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    if movie not in movies['title'].values:
        return jsonify({'status': 'error', 'message': 'Movie not found'})
    
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), 
        reverse=True, 
        key=lambda x: x[1]
    )[1:6]
    recommended_movies = [movies.iloc[i[0]].title for i in distances]
    return jsonify({'status': 'success', 'movies': recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)
