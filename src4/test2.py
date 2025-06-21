#!/usr/bin/env python3
"""
Simplified Movie Recommendation API working with your existing database structure
No external dependencies required except Flask
"""

from flask import Flask, request, jsonify, after_request
import sqlite3
import json
import logging
from datetime import datetime, timedelta
import time
import os

app = Flask(__name__)

# Simple CORS handling without flask_cors
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Configuration - Using SQLite for simplicity (can be changed to PostgreSQL later)
DATABASE_PATH = 'movie_recommendations.db'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDatabaseManager:
    """Simple database manager using SQLite (matches your existing structure)"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def init_database(self):
        """Initialize database with your existing structure"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create tables matching your existing PostgreSQL structure
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                display_name TEXT,
                avatar_url TEXT,
                preferences TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watched_movies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER,
                tmdb_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                rating REAL CHECK (rating >= 0 AND rating <= 10),
                current_mood TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mood_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                mood TEXT NOT NULL,
                selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                page TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tmdb_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                vote_average REAL,
                release_date TEXT,
                popularity REAL,
                similarity_score REAL NOT NULL,
                genres TEXT,
                overview TEXT,
                poster_path TEXT,
                recommendation_method TEXT DEFAULT 'content_based',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP DEFAULT (datetime('now', '+24 hours')),
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, tmdb_id)
            )
        ''')
        
        # Insert sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self.insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
    
    def insert_sample_data(self, cursor):
        """Insert sample data for testing"""
        # Sample users
        users = [
            ('john_doe', 'john@example.com', 'John Doe'),
            ('jane_smith', 'jane@example.com', 'Jane Smith'),
            ('mike_wilson', 'mike@example.com', 'Mike Wilson'),
        ]
        
        for username, email, display_name in users:
            cursor.execute(
                "INSERT INTO users (username, email, display_name) VALUES (?, ?, ?)",
                (username, email, display_name)
            )
        
        # Sample watched movies (using real TMDB IDs)
        watched_movies = [
            (1, 550, 'Fight Club', 9.0, 'excited'),
            (1, 13, 'Forrest Gump', 8.5, 'neutral'),
            (1, 680, 'Pulp Fiction', 9.5, 'excited'),
            (2, 19404, 'Dilwale Dulhania Le Jayenge', 8.0, 'happy'),
            (2, 597, 'Titanic', 7.5, 'sad'),
            (3, 11, 'Star Wars', 8.8, 'excited'),
            (3, 157336, 'Interstellar', 9.2, 'neutral'),
        ]
        
        for user_id, tmdb_id, title, rating, mood in watched_movies:
            cursor.execute('''
                INSERT INTO watched_movies (user_id, tmdb_id, title, rating, current_mood)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, tmdb_id, title, rating, mood))
    
    def get_user_data_for_recommendation(self, user_id):
        """Get user data formatted for recommendation"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Get user info
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return {}
            
            # Get user's movie history
            cursor.execute('''
                SELECT tmdb_id, rating FROM watched_movies 
                WHERE user_id = ? ORDER BY watched_at DESC
            ''', (user_id,))
            
            history_data = cursor.fetchall()
            user_history = [row['tmdb_id'] for row in history_data]
            ratings = {str(row['tmdb_id']): row['rating'] for row in history_data if row['rating']}
            
            # Get latest mood
            cursor.execute('''
                SELECT mood FROM mood_selections 
                WHERE user_id = ? ORDER BY selected_at DESC LIMIT 1
            ''', (user_id,))
            
            mood_row = cursor.fetchone()
            mood = mood_row['mood'] if mood_row else 'neutral'
            
            conn.close()
            
            return {
                'user_id': user_id,
                'username': user['username'],
                'user_history': user_history,
                'ratings': ratings,
                'favourite_genres': [],  # Can be extracted from preferences JSON
                'mood': mood,
                'time_watched': 'day',
                'count': 10
            }
            
        except Exception as e:
            logger.error(f"Error getting user data: {e}")
            return {}
    
    def store_recommendations(self, user_id, recommendations):
        """Store recommendations in database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Clear old recommendations
            cursor.execute("DELETE FROM user_recommendations WHERE user_id = ?", (user_id,))
            
            # Insert new recommendations
            for rec in recommendations:
                cursor.execute('''
                    INSERT INTO user_recommendations 
                    (user_id, tmdb_id, title, vote_average, release_date, popularity, 
                     similarity_score, genres, overview, poster_path, recommendation_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    rec.get('tmdb_id', 0),
                    rec.get('title', ''),
                    rec.get('vote_average'),
                    rec.get('release_date'),
                    rec.get('popularity'),
                    rec.get('similarity_score', 0.0),
                    json.dumps(rec.get('genres', [])),
                    rec.get('overview'),
                    rec.get('poster_path'),
                    rec.get('recommendation_method', 'content_based')
                ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
            return False
    
    def get_stored_recommendations(self, user_id, limit=10):
        """Get stored recommendations from database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_recommendations 
                WHERE user_id = ? AND is_active = 1 AND expires_at > datetime('now')
                ORDER BY similarity_score DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            recommendations = cursor.fetchall()
            conn.close()
            
            result = []
            for rec in recommendations:
                rec_dict = dict(rec)
                rec_dict['genres'] = json.loads(rec_dict.get('genres', '[]'))
                result.append(rec_dict)
            
            return result
        except Exception as e:
            logger.error(f"Error getting stored recommendations: {e}")
            return []

# Simple content-based recommendation system (no external ML libraries)
class SimpleRecommendationEngine:
    """Simple recommendation engine using basic similarity"""
    
    def __init__(self):
        # Popular movies with genres (TMDB data)
        self.popular_movies = [
            {
                'tmdb_id': 278, 'title': 'The Shawshank Redemption',
                'genres': ['Drama'], 'vote_average': 8.7, 'popularity': 75.0,
                'release_date': '1994-09-23', 'overview': 'Two imprisoned men bond over a number of years...'
            },
            {
                'tmdb_id': 238, 'title': 'The Godfather',
                'genres': ['Drama', 'Crime'], 'vote_average': 8.7, 'popularity': 70.0,
                'release_date': '1972-03-14', 'overview': 'The aging patriarch of an organized crime dynasty...'
            },
            {
                'tmdb_id': 424, 'title': 'Schindler\'s List',
                'genres': ['Drama', 'History', 'War'], 'vote_average': 8.6, 'popularity': 68.0,
                'release_date': '1993-12-15', 'overview': 'The true story of how businessman Oskar Schindler...'
            },
            {
                'tmdb_id': 389, 'title': 'The Matrix',
                'genres': ['Action', 'Science Fiction'], 'vote_average': 8.2, 'popularity': 85.0,
                'release_date': '1999-03-30', 'overview': 'Set in the 22nd century, The Matrix tells the story...'
            },
            {
                'tmdb_id': 155, 'title': 'The Dark Knight',
                'genres': ['Action', 'Crime', 'Drama'], 'vote_average': 8.5, 'popularity': 90.0,
                'release_date': '2008-07-16', 'overview': 'Batman raises the stakes in his war on crime...'
            }
        ]
    
    def recommend(self, user_data):
        """Simple recommendation algorithm"""
        user_history = user_data.get('user_history', [])
        ratings = user_data.get('ratings', {})
        mood = user_data.get('mood', 'neutral')
        count = user_data.get('count', 10)
        
        # Simple scoring: avoid already watched movies, prefer higher rated movies
        recommendations = []
        
        for movie in self.popular_movies:
            if movie['tmdb_id'] not in user_history:
                # Simple similarity score based on rating and popularity
                similarity_score = (movie['vote_average'] / 10.0) * 0.7 + (movie['popularity'] / 100.0) * 0.3
                
                # Mood adjustment
                if mood == 'happy' and any(genre in ['Comedy', 'Family'] for genre in movie['genres']):
                    similarity_score *= 1.2
                elif mood == 'excited' and any(genre in ['Action', 'Adventure'] for genre in movie['genres']):
                    similarity_score *= 1.2
                elif mood == 'sad' and any(genre in ['Drama', 'Romance'] for genre in movie['genres']):
                    similarity_score *= 1.1
                
                recommendations.append({
                    'tmdb_id': movie['tmdb_id'],
                    'title': movie['title'],
                    'vote_average': movie['vote_average'],
                    'release_date': movie['release_date'],
                    'popularity': movie['popularity'],
                    'similarity_score': round(min(similarity_score, 1.0), 4),
                    'genres': movie['genres'],
                    'overview': movie['overview'],
                    'recommendation_method': 'content_based'
                })
        
        # Sort by similarity score and return top recommendations
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations[:count]

# Initialize components
db_manager = SimpleDatabaseManager(DATABASE_PATH)
recommendation_engine = SimpleRecommendationEngine()

# API Routes
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Simple Movie Recommendation API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/users/<int:user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    """Get recommendations for a user"""
    try:
        # Check for stored recommendations first
        stored_recs = db_manager.get_stored_recommendations(user_id)
        
        if stored_recs:
            return jsonify({
                'user_id': user_id,
                'recommendations': stored_recs,
                'count': len(stored_recs),
                'source': 'cached'
            })
        
        # Generate fresh recommendations
        user_data = db_manager.get_user_data_for_recommendation(user_id)
        if not user_data:
            return jsonify({'error': 'User not found or no movie history'}), 404
        
        recommendations = recommendation_engine.recommend(user_data)
        
        # Store recommendations
        if recommendations:
            db_manager.store_recommendations(user_id, recommendations)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations),
            'source': 'generated'
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/users/<int:user_id>/movies', methods=['POST'])
def add_movie_to_history(user_id):
    """Add a movie to user's watch history"""
    try:
        data = request.get_json()
        tmdb_id = data.get('tmdb_id')
        title = data.get('title')
        rating = data.get('rating')
        mood = data.get('mood', 'neutral')
        
        if not tmdb_id or not title:
            return jsonify({'error': 'tmdb_id and title are required'}), 400
        
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO watched_movies (user_id, tmdb_id, title, rating, current_mood)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, tmdb_id, title, rating, mood))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Movie added to history successfully',
            'user_id': user_id,
            'tmdb_id': tmdb_id,
            'title': title,
            'rating': rating
        })
        
    except Exception as e:
        logger.error(f"Error adding movie to history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/users/<int:user_id>/mood', methods=['POST'])
def set_user_mood(user_id):
    """Set user's current mood"""
    try:
        data = request.get_json()
        mood = data.get('mood', 'neutral')
        page = data.get('page', 'api')
        
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mood_selections (user_id, mood, page)
            VALUES (?, ?, ?)
        ''', (user_id, mood, page))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Mood updated successfully',
            'user_id': user_id,
            'mood': mood
        })
        
    except Exception as e:
        logger.error(f"Error setting mood: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/users/<int:user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    """Get user profile and statistics"""
    try:
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Get user info
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) as total_movies FROM watched_movies WHERE user_id = ?", (user_id,))
        total_movies = cursor.fetchone()['total_movies']
        
        cursor.execute("SELECT AVG(rating) as avg_rating FROM watched_movies WHERE user_id = ? AND rating IS NOT NULL", (user_id,))
        avg_rating_result = cursor.fetchone()
        avg_rating = avg_rating_result['avg_rating'] if avg_rating_result['avg_rating'] else 0
        
        conn.close()
        
        return jsonify({
            'user_id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'display_name': user['display_name'],
            'total_movies_watched': total_movies,
            'average_rating': round(avg_rating, 2) if avg_rating else None,
            'created_at': user['created_at']
        })
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Simple Movie Recommendation API...")
    print(f"📊 Database: {DATABASE_PATH}")
    print("🌐 API will be available at: http://localhost:5000")
    print("\n📝 Sample API calls:")
    print("GET  http://localhost:5000/users/1/recommendations")
    print("POST http://localhost:5000/users/1/movies")
    print("GET  http://localhost:5000/users/1/profile")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
