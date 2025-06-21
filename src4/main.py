#!/usr/bin/env python3
"""
Hybrid Movie Recommendation Engine with Collaborative Filtering
Handles Cold Start with Content-Based, switches to Collaborative Filtering when data is sufficient
"""

import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd

class CollaborativeFilteringEngine:
    """Implements User-User and Item-Item Collaborative Filtering"""
    
    def __init__(self, min_common_items: int = 3, min_common_users: int = 3):
        self.min_common_items = min_common_items  # Minimum items in common for user similarity
        self.min_common_users = min_common_users  # Minimum users in common for item similarity
        self.user_similarity_cache = {}
        self.item_similarity_cache = {}
        
    def build_rating_matrix(self, all_user_data: List[Dict]) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Build user-item rating matrix from database data"""
        ratings_data = []
        
        for user_data in all_user_data:
            user_id = user_data['user_id']
            ratings = user_data.get('ratings', {})
            
            for movie_id_str, rating in ratings.items():
                ratings_data.append({
                    'user_id': user_id,
                    'movie_id': int(movie_id_str),
                    'rating': float(rating)
                })
        
        if not ratings_data:
            return pd.DataFrame(), {}, {}
        
        # Create rating matrix
        rating_df = pd.DataFrame(ratings_data)
        rating_matrix = rating_df.pivot(index='user_id', columns='movie_id', values='rating')
        rating_matrix = rating_matrix.fillna(0)  # Fill NaN with 0
        
        # Create mappings
        user_to_idx = {user_id: idx for idx, user_id in enumerate(rating_matrix.index)}
        item_to_idx = {movie_id: idx for idx, movie_id in enumerate(rating_matrix.columns)}
        
        return rating_matrix, user_to_idx, item_to_idx
    
    def calculate_user_similarity(self, rating_matrix: pd.DataFrame, user_id: int, target_user_id: int) -> float:
        """Calculate similarity between two users using Pearson correlation"""
        if user_id == target_user_id:
            return 1.0
        
        # Get user ratings
        user1_ratings = rating_matrix.loc[user_id]
        user2_ratings = rating_matrix.loc[target_user_id]
        
        # Find commonly rated items
        common_items = (user1_ratings > 0) & (user2_ratings > 0)
        
        if common_items.sum() < self.min_common_items:
            return 0.0
        
        user1_common = user1_ratings[common_items]
        user2_common = user2_ratings[common_items]
        
        # Calculate Pearson correlation
        try:
            correlation, _ = pearsonr(user1_common, user2_common)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def calculate_item_similarity(self, rating_matrix: pd.DataFrame, item_id: int, target_item_id: int) -> float:
        """Calculate similarity between two items using cosine similarity"""
        if item_id == target_item_id:
            return 1.0
        
        # Get item ratings
        item1_ratings = rating_matrix[item_id]
        item2_ratings = rating_matrix[target_item_id]
        
        # Find users who rated both items
        common_users = (item1_ratings > 0) & (item2_ratings > 0)
        
        if common_users.sum() < self.min_common_users:
            return 0.0
        
        item1_common = item1_ratings[common_users].values.reshape(1, -1)
        item2_common = item2_ratings[common_users].values.reshape(1, -1)
        
        # Calculate cosine similarity
        try:
            similarity = cosine_similarity(item1_common, item2_common)[0][0]
            return similarity if not np.isnan(similarity) else 0.0
        except:
            return 0.0
    
    def get_user_based_recommendations(self, rating_matrix: pd.DataFrame, user_id: int, 
                                     candidate_movies: List[int], k_neighbors: int = 20) -> Dict[int, float]:
        """Generate recommendations using User-User Collaborative Filtering"""
        if user_id not in rating_matrix.index:
            return {}
        
        # Find similar users
        user_similarities = []
        target_user_ratings = rating_matrix.loc[user_id]
        
        for other_user_id in rating_matrix.index:
            if other_user_id != user_id:
                similarity = self.calculate_user_similarity(rating_matrix, user_id, other_user_id)
                if similarity > 0:
                    user_similarities.append((other_user_id, similarity))
        
        # Sort by similarity and take top k
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = user_similarities[:k_neighbors]
        
        if not top_similar_users:
            return {}
        
        # Calculate predicted ratings for candidate movies
        predictions = {}
        
        for movie_id in candidate_movies:
            if movie_id in rating_matrix.columns and target_user_ratings[movie_id] == 0:  # User hasn't rated this movie
                numerator = 0
                denominator = 0
                
                for similar_user_id, similarity in top_similar_users:
                    similar_user_rating = rating_matrix.loc[similar_user_id, movie_id]
                    
                    if similar_user_rating > 0:  # Similar user has rated this movie
                        # Use mean-centered ratings
                        similar_user_mean = rating_matrix.loc[similar_user_id][rating_matrix.loc[similar_user_id] > 0].mean()
                        numerator += similarity * (similar_user_rating - similar_user_mean)
                        denominator += abs(similarity)
                
                if denominator > 0:
                    user_mean = target_user_ratings[target_user_ratings > 0].mean()
                    predicted_rating = user_mean + (numerator / denominator)
                    predictions[movie_id] = max(0, min(10, predicted_rating))  # Clamp between 0-10
        
        return predictions
    
    def get_item_based_recommendations(self, rating_matrix: pd.DataFrame, user_id: int, 
                                     candidate_movies: List[int], k_neighbors: int = 20) -> Dict[int, float]:
        """Generate recommendations using Item-Item Collaborative Filtering"""
        if user_id not in rating_matrix.index:
            return {}
        
        user_ratings = rating_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_movies:
            return {}
        
        predictions = {}
        
        for movie_id in candidate_movies:
            if movie_id not in rating_matrix.columns or user_ratings[movie_id] > 0:
                continue  # Skip if movie not in matrix or already rated
            
            # Find similar items to this movie
            item_similarities = []
            
            for rated_movie_id in rated_movies:
                similarity = self.calculate_item_similarity(rating_matrix, movie_id, rated_movie_id)
                if similarity > 0:
                    item_similarities.append((rated_movie_id, similarity))
            
            # Sort by similarity and take top k
            item_similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar_items = item_similarities[:k_neighbors]
            
            if not top_similar_items:
                continue
            
            # Calculate predicted rating
            numerator = 0
            denominator = 0
            
            for similar_item_id, similarity in top_similar_items:
                user_rating_for_similar_item = user_ratings[similar_item_id]
                numerator += similarity * user_rating_for_similar_item
                denominator += abs(similarity)
            
            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions[movie_id] = max(0, min(10, predicted_rating))
        
        return predictions


class HybridRecommendationEngine:
    """Hybrid system that switches between Content-Based and Collaborative Filtering"""
    
    def __init__(self, tmdb_api_key: str):
        self.tmdb_api_key = tmdb_api_key
        self.content_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collaborative_engine = CollaborativeFilteringEngine()
        
        # Cold start thresholds
        self.min_users_for_cf = 10          # Minimum users needed for collaborative filtering
        self.min_ratings_per_user = 5       # Minimum ratings per user for collaborative filtering
        self.min_total_ratings = 50         # Minimum total ratings in system
        
        # Genre mapping
        self.genre_mapping = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }
        
        print("✅ Hybrid Recommendation Engine initialized")
    
    def get_recommendations(self, 
                          user_data: Dict,
                          all_user_data: List[Dict],
                          mood: str = "neutral",
                          count: int = 10,
                          time_watched: str = "day") -> List[Dict]:
        """
        Main recommendation function with hybrid approach
        
        Args:
            user_data: Current user's data
            all_user_data: All users' data from database for collaborative filtering
            mood: User's current mood
            count: Number of recommendations
            time_watched: When user watches movies
            
        Returns:
            List of movie recommendations
        """
        try:
            print(f"🤖 Generating hybrid recommendations...")
            
            # Assess system state for collaborative filtering readiness
            cf_readiness = self._assess_collaborative_filtering_readiness(user_data, all_user_data)
            
            user_history = user_data.get('user_history', [])
            ratings = user_data.get('ratings', {})
            favourite_genres = user_data.get('favourite_genres', [])
            
            print(f"📊 CF Readiness: {cf_readiness}")
            
            # Get candidate movies from TMDB
            candidate_movies = self._fetch_candidate_movies()
            candidate_movie_ids = [movie['id'] for movie in candidate_movies]
            
            if cf_readiness['use_collaborative']:
                print("🔄 Using Collaborative Filtering (sufficient data available)")
                recommendations = self._get_collaborative_recommendations(
                    user_data, all_user_data, candidate_movies, count
                )
            else:
                print("🎯 Using Content-Based Filtering (cold start or insufficient data)")
                recommendations = self._get_content_based_recommendations(
                    user_history, ratings, favourite_genres, mood, time_watched, candidate_movies, count
                )
            
            # If collaborative filtering doesn't return enough results, fall back to content-based
            if len(recommendations) < count and cf_readiness['use_collaborative']:
                print("🔄 Collaborative filtering insufficient, adding content-based recommendations")
                remaining_count = count - len(recommendations)
                content_recommendations = self._get_content_based_recommendations(
                    user_history, ratings, favourite_genres, mood, time_watched, candidate_movies, remaining_count
                )
                
                # Merge recommendations, avoiding duplicates
                existing_ids = {rec['tmdb_id'] for rec in recommendations}
                for rec in content_recommendations:
                    if rec['tmdb_id'] not in existing_ids and len(recommendations) < count:
                        recommendations.append(rec)
            
            print(f"✅ Generated {len(recommendations)} hybrid recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in hybrid recommendations: {e}")
            # Fall back to content-based
            return self._get_content_based_recommendations(
                user_history, ratings, favourite_genres, mood, time_watched, 
                self._fetch_candidate_movies(), count
            )
    
    def _assess_collaborative_filtering_readiness(self, user_data: Dict, all_user_data: List[Dict]) -> Dict:
        """Assess whether the system has enough data for collaborative filtering"""
        total_users = len(all_user_data)
        total_ratings = sum(len(user.get('ratings', {})) for user in all_user_data)
        
        current_user_ratings = len(user_data.get('ratings', {}))
        
        # Users with sufficient ratings
        users_with_enough_ratings = sum(
            1 for user in all_user_data 
            if len(user.get('ratings', {})) >= self.min_ratings_per_user
        )
        
        # Decision logic
        use_collaborative = (
            total_users >= self.min_users_for_cf and
            total_ratings >= self.min_total_ratings and
            users_with_enough_ratings >= (self.min_users_for_cf * 0.5) and
            current_user_ratings >= self.min_ratings_per_user
        )
        
        return {
            'use_collaborative': use_collaborative,
            'total_users': total_users,
            'total_ratings': total_ratings,
            'current_user_ratings': current_user_ratings,
            'users_with_enough_ratings': users_with_enough_ratings,
            'reason': self._get_cf_decision_reason(use_collaborative, total_users, total_ratings, 
                                                 current_user_ratings, users_with_enough_ratings)
        }
    
    def _get_cf_decision_reason(self, use_cf: bool, total_users: int, total_ratings: int, 
                               current_user_ratings: int, users_with_enough_ratings: int) -> str:
        """Get human-readable reason for CF decision"""
        if use_cf:
            return f"Sufficient data: {total_users} users, {total_ratings} ratings, user has {current_user_ratings} ratings"
        
        reasons = []
        if total_users < self.min_users_for_cf:
            reasons.append(f"Need {self.min_users_for_cf} users (have {total_users})")
        if total_ratings < self.min_total_ratings:
            reasons.append(f"Need {self.min_total_ratings} total ratings (have {total_ratings})")
        if current_user_ratings < self.min_ratings_per_user:
            reasons.append(f"User needs {self.min_ratings_per_user} ratings (has {current_user_ratings})")
        if users_with_enough_ratings < (self.min_users_for_cf * 0.5):
            reasons.append(f"Need more active users with {self.min_ratings_per_user}+ ratings")
        
        return "Cold start: " + ", ".join(reasons)
    
    def _get_collaborative_recommendations(self, user_data: Dict, all_user_data: List[Dict], 
                                         candidate_movies: List[Dict], count: int) -> List[Dict]:
        """Get recommendations using collaborative filtering"""
        try:
            # Build rating matrix
            rating_matrix, user_to_idx, item_to_idx = self.collaborative_engine.build_rating_matrix(all_user_data)
            
            if rating_matrix.empty:
                print("❌ Empty rating matrix, falling back to content-based")
                return []
            
            user_id = user_data['user_id']
            candidate_movie_ids = [movie['id'] for movie in candidate_movies]
            
            # Get predictions from both approaches
            user_based_predictions = self.collaborative_engine.get_user_based_recommendations(
                rating_matrix, user_id, candidate_movie_ids, k_neighbors=15
            )
            
            item_based_predictions = self.collaborative_engine.get_item_based_recommendations(
                rating_matrix, user_id, candidate_movie_ids, k_neighbors=15
            )
            
            # Combine predictions (weighted average: 60% user-based, 40% item-based)
            combined_predictions = {}
            all_predicted_items = set(user_based_predictions.keys()) | set(item_based_predictions.keys())
            
            for movie_id in all_predicted_items:
                user_score = user_based_predictions.get(movie_id, 0)
                item_score = item_based_predictions.get(movie_id, 0)
                
                if user_score > 0 and item_score > 0:
                    # Both methods have predictions
                    combined_score = 0.6 * user_score + 0.4 * item_score
                elif user_score > 0:
                    # Only user-based has prediction
                    combined_score = user_score * 0.8  # Slightly reduce confidence
                elif item_score > 0:
                    # Only item-based has prediction
                    combined_score = item_score * 0.8
                else:
                    combined_score = 0
                
                combined_predictions[movie_id] = combined_score
            
            # Sort by predicted rating
            sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Build recommendation list
            recommendations = []
            watched_movies = set(user_data.get('user_history', []))
            
            for movie_id, predicted_rating in sorted_predictions:
                if movie_id not in watched_movies and len(recommendations) < count:
                    # Find movie details
                    movie_details = next((m for m in candidate_movies if m['id'] == movie_id), None)
                    if movie_details:
                        recommendation = {
                            'tmdb_id': movie_details['id'],
                            'title': movie_details['title'],
                            'vote_average': movie_details.get('vote_average', 0.0),
                            'release_date': movie_details.get('release_date', ''),
                            'popularity': movie_details.get('popularity', 0.0),
                            'similarity_score': round(predicted_rating / 10.0, 4),  # Normalize to 0-1
                            'genres': [g['name'] for g in movie_details.get('genres', [])],
                            'overview': movie_details.get('overview', ''),
                            'recommendation_method': 'collaborative_filtering'
                        }
                        recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in collaborative filtering: {e}")
            return []
    
    def _get_content_based_recommendations(self, user_history: List[int], ratings: Dict[str, float], 
                                         favourite_genres: List[str], mood: str, time_watched: str,
                                         candidate_movies: List[Dict], count: int) -> List[Dict]:
        """Get recommendations using content-based filtering (existing logic)"""
        try:
            user_preferences = self._extract_user_preferences(
                user_history, ratings, favourite_genres, mood, time_watched
            )
            
            # Score all candidates
            scored_movies = []
            for movie in candidate_movies:
                score = self._score_movie(movie, user_preferences)
                
                recommendation = {
                    'tmdb_id': movie['id'],
                    'title': movie['title'],
                    'vote_average': movie.get('vote_average', 0.0),
                    'release_date': movie.get('release_date', ''),
                    'popularity': movie.get('popularity', 0.0),
                    'similarity_score': round(score, 4),
                    'genres': [g['name'] for g in movie.get('genres', [])],
                    'overview': movie.get('overview', ''),
                    'recommendation_method': 'content_based'
                }
                scored_movies.append(recommendation)
            
            # Filter out already watched movies
            watched_ids = set(user_history)
            filtered_movies = [movie for movie in scored_movies if movie['tmdb_id'] not in watched_ids]
            
            # Sort by similarity score and return top recommendations
            filtered_movies.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return filtered_movies[:count]
            
        except Exception as e:
            print(f"❌ Error in content-based filtering: {e}")
            return []
    
    def _extract_user_preferences(self, user_history: List[int], ratings: Dict[str, float] = None,
                                 favourite_genres: List[str] = None, mood: str = "neutral",
                                 time_watched: str = "day") -> Dict:
        """Extract comprehensive user preferences (existing logic from previous code)"""
        # Fetch user's movie history details from TMDB
        history_movies = self._fetch_movie_details(user_history)
        
        if not history_movies:
            return self._get_default_preferences()
        
        # Extract various preference dimensions
        genre_preferences = self._extract_genre_preferences(history_movies, ratings, favourite_genres)
        quality_preferences = self._extract_quality_preferences(history_movies, ratings)
        content_preferences = self._extract_content_preferences(history_movies)
        temporal_preferences = self._extract_temporal_preferences(history_movies)
        
        # Add mood and time preferences
        mood_adjustment = self._get_mood_adjustment(mood)
        time_adjustment = self._get_time_adjustment(time_watched)
        
        return {
            'genre_vector': genre_preferences,
            'quality_profile': quality_preferences,
            'content_embedding': content_preferences,
            'temporal_pattern': temporal_preferences,
            'mood_adjustment': mood_adjustment,
            'time_preference': time_adjustment,
            'confidence_score': min(len(history_movies) / 10.0, 1.0)
        }
    
    # Include all the existing content-based methods from previous code
    def _fetch_movie_details(self, movie_ids: List[int]) -> List[Dict]:
        """Fetch movie details from TMDB API"""
        movies = []
        for movie_id in movie_ids:
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}",
                    params={"api_key": self.tmdb_api_key},
                    timeout=5
                )
                if response.status_code == 200:
                    movies.append(response.json())
            except Exception as e:
                print(f"Failed to fetch movie {movie_id}: {e}")
                continue
        return movies
    
    def _extract_genre_preferences(self, movies: List[Dict], ratings: Dict[str, float] = None,
                                  favourite_genres: List[str] = None) -> np.ndarray:
        """Create weighted genre preference vector"""
        genre_weights = Counter()
        
        # Weight from movie history
        for movie in movies:
            movie_rating = 7.0  # Default
            if ratings and str(movie['id']) in ratings:
                movie_rating = ratings[str(movie['id'])]
            elif 'vote_average' in movie:
                movie_rating = movie['vote_average']
            
            weight = max(movie_rating / 10.0, 0.1)
            
            for genre in movie.get('genres', []):
                genre_name = genre['name']
                genre_weights[genre_name] += weight
        
        # Add favourite genres with extra weight
        if favourite_genres:
            for genre in favourite_genres:
                genre_weights[genre] += 1.5
        
        # Create normalized vector
        genre_vector = np.zeros(len(self.genre_mapping))
        total_weight = sum(genre_weights.values()) if genre_weights else 1
        
        for i, genre_name in enumerate(self.genre_mapping.values()):
            if genre_name in genre_weights:
                genre_vector[i] = genre_weights[genre_name] / total_weight
        
        return genre_vector
    
    def _extract_quality_preferences(self, movies: List[Dict], ratings: Dict[str, float] = None) -> Dict:
        """Extract quality and rating preferences"""
        movie_ratings = []
        popularities = []
        
        for movie in movies:
            if ratings and str(movie['id']) in ratings:
                movie_ratings.append(ratings[str(movie['id'])])
            else:
                movie_ratings.append(movie.get('vote_average', 7.0))
            
            popularities.append(movie.get('popularity', 50.0))
        
        return {
            'min_rating': np.percentile(movie_ratings, 25) if movie_ratings else 6.5,
            'avg_rating': np.mean(movie_ratings) if movie_ratings else 7.0,
            'popularity_pref': np.mean(popularities) if popularities else 50.0,
            'quality_threshold': np.median(movie_ratings) if movie_ratings else 7.0
        }
    
    def _extract_content_preferences(self, movies: List[Dict]) -> np.ndarray:
        """Create content-based preference embedding"""
        try:
            texts = []
            for movie in movies:
                text = f"{movie['title']}. {movie.get('overview', '')}"
                texts.append(text)
            
            if texts:
                embeddings = self.content_model.encode(texts)
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(384)
        except Exception as e:
            print(f"Content preference extraction error: {e}")
            return np.zeros(384)
    
    def _extract_temporal_preferences(self, movies: List[Dict]) -> Dict:
        """Extract temporal/era preferences"""
        years = []
        for movie in movies:
            release_date = movie.get('release_date', '')
            if release_date:
                try:
                    year = int(release_date[:4])
                    years.append(year)
                except ValueError:
                    continue
        
        if years:
            return {
                'preferred_era': np.mean(years),
                'era_variance': np.std(years),
                'modern_bias': sum(1 for y in years if y >= 2015) / len(years)
            }
        else:
            return {'preferred_era': 2020, 'era_variance': 8, 'modern_bias': 0.7}
    
    def _get_mood_adjustment(self, mood: str) -> Dict:
        """Get mood-based preference adjustments"""
        mood_mappings = {
            "happy": {"Comedy": 1.3, "Adventure": 1.2, "Family": 1.1},
            "sad": {"Drama": 1.3, "Romance": 1.2, "Documentary": 1.1},
            "excited": {"Action": 1.4, "Thriller": 1.3, "Science Fiction": 1.2},
            "neutral": {}
        }
        return mood_mappings.get(mood, {})
    
    def _get_time_adjustment(self, time_watched: str) -> Dict:
        """Get time-based preference adjustments"""
        time_mappings = {
            "night": {"Horror": 1.2, "Thriller": 1.2, "Mystery": 1.1},
            "day": {"Comedy": 1.1, "Family": 1.1, "Documentary": 1.1}
        }
        return time_mappings.get(time_watched, {})
    
    def _score_movie(self, movie: Dict, user_preferences: Dict) -> float:
        """Score a movie against user preferences"""
        try:
            # Extract movie features
            movie_features = self._extract_movie_features(movie)
            
            # Calculate individual scores
            genre_score = self._calculate_genre_similarity(
                movie_features['genre_vector'], 
                user_preferences['genre_vector']
            )
            
            content_score = self._calculate_content_similarity(
                movie_features['content_embedding'], 
                user_preferences['content_embedding']
            )
            
            quality_score = self._calculate_quality_match(
                movie_features['quality_metrics'], 
                user_preferences['quality_profile']
            )
            
            temporal_score = self._calculate_temporal_match(
                movie_features['temporal_info'], 
                user_preferences['temporal_pattern']
            )
            
            # Apply mood and time adjustments
            mood_boost = self._apply_mood_adjustment(
                movie, user_preferences['mood_adjustment']
            )
            time_boost = self._apply_time_adjustment(
                movie, user_preferences['time_preference']
            )
            
            # Weighted combination
            base_score = (
                0.35 * genre_score +
                0.25 * content_score +
                0.25 * quality_score +
                0.15 * temporal_score
            )
            
            # Apply adjustments
            final_score = base_score * (1 + mood_boost + time_boost)
            
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error scoring movie: {e}")
            return 0.0
    
    def _extract_movie_features(self, movie: Dict) -> Dict:
        """Extract features from a movie"""
        # Genre vector
        genre_vector = np.zeros(len(self.genre_mapping))
        for genre in movie.get('genres', []):
            genre_name = genre['name'] if isinstance(genre, dict) else genre
            try:
                genre_idx = list(self.genre_mapping.values()).index(genre_name)
                genre_vector[genre_idx] = 1.0
            except ValueError:
                continue
        
        # Content embedding
        text = f"{movie['title']}. {movie.get('overview', '')}"
        try:
            content_embedding = self.content_model.encode([text])[0]
        except:
            content_embedding = np.zeros(384)
        
        # Quality metrics
        quality_metrics = {
            'rating': movie.get('vote_average', 0),
            'popularity': movie.get('popularity', 0),
            'vote_count': movie.get('vote_count', 0)
        }
        
        # Temporal info
        temporal_info = {}
        release_date = movie.get('release_date', '')
        if release_date:
            try:
                temporal_info['year'] = int(release_date[:4])
            except ValueError:
                temporal_info['year'] = 2020
        else:
            temporal_info['year'] = 2020
        
        return {
            'genre_vector': genre_vector,
            'content_embedding': content_embedding,
            'quality_metrics': quality_metrics,
            'temporal_info': temporal_info
        }
    
    def _calculate_genre_similarity(self, movie_genres: np.ndarray, user_genres: np.ndarray) -> float:
        """Calculate genre similarity using cosine similarity"""
        if np.sum(user_genres) == 0 or np.sum(movie_genres) == 0:
            return 0.5
        
        norm_product = np.linalg.norm(movie_genres) * np.linalg.norm(user_genres)
        if norm_product == 0:
            return 0.5
        
        return np.dot(movie_genres, user_genres) / norm_product
    
    def _calculate_content_similarity(self, movie_embedding: np.ndarray, user_embedding: np.ndarray) -> float:
        """Calculate content similarity using cosine similarity"""
        if np.sum(user_embedding) == 0 or np.sum(movie_embedding) == 0:
            return 0.5
        
        norm_product = np.linalg.norm(movie_embedding) * np.linalg.norm(user_embedding)
        if norm_product == 0:
            return 0.5
        
        return np.dot(movie_embedding, user_embedding) / norm_product
    
    def _calculate_quality_match(self, movie_quality: Dict, user_quality: Dict) -> float:
        """Calculate quality preference match"""
        movie_rating = movie_quality['rating']
        user_threshold = user_quality['quality_threshold']
        
        if movie_rating < user_threshold - 1.0:
            return 0.2
        
        rating_diff = abs(movie_rating - user_quality['avg_rating'])
        return max(0, 1 - (rating_diff / 5.0))
    
    def _calculate_temporal_match(self, movie_temporal: Dict, user_temporal: Dict) -> float:
        """Calculate temporal preference match"""
        movie_year = movie_temporal['year']
        user_era = user_temporal['preferred_era']
        user_variance = user_temporal['era_variance']
        
        year_diff = abs(movie_year - user_era)
        temporal_score = max(0, 1 - (year_diff / (user_variance * 2)))
        
        return temporal_score
    
    def _apply_mood_adjustment(self, movie: Dict, mood_adjustments: Dict) -> float:
        """Apply mood-based score adjustments"""
        boost = 0.0
        for genre in movie.get('genres', []):
            genre_name = genre['name'] if isinstance(genre, dict) else genre
            if genre_name in mood_adjustments:
                boost += (mood_adjustments[genre_name] - 1.0) * 0.1
        return boost
    
    def _apply_time_adjustment(self, movie: Dict, time_adjustments: Dict) -> float:
        """Apply time-based score adjustments"""
        boost = 0.0
        for genre in movie.get('genres', []):
            genre_name = genre['name'] if isinstance(genre, dict) else genre
            if genre_name in time_adjustments:
                boost += (time_adjustments[genre_name] - 1.0) * 0.05
        return boost
    
    def _fetch_candidate_movies(self) -> List[Dict]:
        """Fetch candidate movies from TMDB for recommendations"""
        candidates = []
        endpoints = [
            "https://api.themoviedb.org/3/movie/popular",
            "https://api.themoviedb.org/3/movie/top_rated",
            "https://api.themoviedb.org/3/discover/movie"
        ]
        
        for endpoint in endpoints:
            try:
                for page in range(1, 4):  # 3 pages per endpoint
                    response = requests.get(
                        endpoint,
                        params={"api_key": self.tmdb_api_key, "page": page},
                        timeout=10
                    )
                    if response.status_code == 200:
                        data = response.json()
                        candidates.extend(data.get('results', []))
            except Exception as e:
                print(f"Error fetching from {endpoint}: {e}")
                continue
        
        # Remove duplicates and add genre details
        unique_movies = {}
        for movie in candidates:
            if movie['id'] not in unique_movies:
                # Fetch detailed info with genres
                try:
                    detail_response = requests.get(
                        f"https://api.themoviedb.org/3/movie/{movie['id']}",
                        params={"api_key": self.tmdb_api_key},
                        timeout=5
                    )
                    if detail_response.status_code == 200:
                        detailed_movie = detail_response.json()
                        unique_movies[movie['id']] = detailed_movie
                except:
                    unique_movies[movie['id']] = movie
        
        return list(unique_movies.values())
    
    def _get_default_preferences(self) -> Dict:
        """Default preferences for users with no history"""
        return {
            'genre_vector': np.ones(len(self.genre_mapping)) / len(self.genre_mapping),
            'quality_profile': {
                'min_rating': 6.5,
                'avg_rating': 7.0,
                'popularity_pref': 50.0,
                'quality_threshold': 6.5
            },
            'content_embedding': np.zeros(384),
            'temporal_pattern': {'preferred_era': 2020, 'era_variance': 8, 'modern_bias': 0.7},
            'mood_adjustment': {},
            'time_preference': {},
            'confidence_score': 0.1
        }


# Simple API wrapper for easy deployment
class HybridRecommendationAPI:
    """Simple API wrapper for the hybrid recommendation engine"""
    
    def __init__(self, tmdb_api_key: str):
        self.engine = HybridRecommendationEngine(tmdb_api_key)
    
    def recommend(self, user_data: Dict, all_user_data: List[Dict]) -> List[Dict]:
        """
        Simple API method for getting hybrid recommendations
        
        Args:
            user_data: Current user's data:
                {
                    "user_id": 123,
                    "user_history": [550, 680, 13],
                    "ratings": {"550": 8.5, "680": 7.2},
                    "favourite_genres": ["Action", "Drama"],
                    "mood": "neutral",
                    "time_watched": "day"
                }
            all_user_data: List of all users' data for collaborative filtering
        
        Returns:
            List of movie recommendations with required fields
        """
        return self.engine.get_recommendations(
            user_data=user_data,
            all_user_data=all_user_data,
            mood=user_data.get("mood", "neutral"),
            count=user_data.get("count", 10),
            time_watched=user_data.get("time_watched", "day")
        )
