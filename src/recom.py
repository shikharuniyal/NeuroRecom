import numpy as np
import requests
from datetime import datetime
from typing import Dict, List
from collections import Counter
from sentence_transformers import SentenceTransformer

class FlexibleFeatureExtractor:
    """Extract user preferences without depending on specific movie IDs"""
    
    def __init__(self, tmdb_api_key: str):
        self.tmdb_api_key = tmdb_api_key
        self.genre_mapping = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }

    def extract_user_preferences(self, user_history: List[int]) -> Dict:
        """Extract user preferences from any movie history"""
        try:
            # Fetch movie details for history movies (real-time API calls)
            history_movies = self._fetch_movie_details(user_history)
            
            if not history_movies:
                return self._get_default_preferences()

            # Extract various preferences
            genre_preferences = self._extract_genre_preferences(history_movies)
            rating_preferences = self._extract_rating_preferences(history_movies)
            content_preferences = self._extract_content_preferences(history_movies)
            temporal_preferences = self._extract_temporal_preferences(history_movies)

            return {
                'genre_vector': genre_preferences,
                'rating_profile': rating_preferences,
                'content_embedding': content_preferences,
                'temporal_pattern': temporal_preferences,
                'confidence_score': len(history_movies) / max(len(user_history), 1)
            }
        except Exception as e:
            print(f"Error extracting user preferences: {e}")
            return self._get_default_preferences()

    def _fetch_movie_details(self, movie_ids: List[int]) -> List[Dict]:
        """Fetch movie details from TMDB API in real-time"""
        movies = []
        for movie_id in movie_ids:
            try:
                response = requests.get(
                    f"https://api.themoviedb.org/3/movie/{movie_id}",
                    params={"api_key": self.tmdb_api_key},
                    timeout=5
                )
                if response.status_code == 200:
                    movie_data = response.json()
                    movies.append({
                        'tmdb_id': movie_data['id'],
                        'title': movie_data['title'],
                        'genres': movie_data.get('genres', []),
                        'overview': movie_data.get('overview', ''),
                        'vote_average': movie_data.get('vote_average', 0),
                        'release_date': movie_data.get('release_date', ''),
                        'popularity': movie_data.get('popularity', 0)
                    })
            except Exception as e:
                print(f"Failed to fetch movie {movie_id}: {e}")
                continue
        return movies

    def _extract_genre_preferences(self, movies: List[Dict]) -> np.ndarray:
        """Create weighted genre preference vector"""
        genre_counts = Counter()
        for movie in movies:
            for genre in movie.get('genres', []):
                genre_name = genre['name']
                # Weight by movie rating (higher rated = stronger preference)
                weight = max(movie.get('vote_average', 5) / 10.0, 0.1)
                genre_counts[genre_name] += weight

        # Create normalized genre vector
        genre_vector = np.zeros(len(self.genre_mapping))
        total_weight = sum(genre_counts.values()) if genre_counts else 1
        
        for i, genre_name in enumerate(self.genre_mapping.values()):
            if genre_name in genre_counts:
                genre_vector[i] = genre_counts[genre_name] / total_weight
        
        return genre_vector

    def _extract_rating_preferences(self, movies: List[Dict]) -> Dict:
        """Extract user's rating and quality preferences"""
        ratings = [movie.get('vote_average', 0) for movie in movies]
        popularities = [movie.get('popularity', 0) for movie in movies]
        
        return {
            'preferred_rating_min': np.percentile(ratings, 25) if ratings else 6.0,
            'preferred_rating_avg': np.mean(ratings) if ratings else 7.0,
            'popularity_preference': np.mean(popularities) if popularities else 50.0,
            'quality_threshold': np.median(ratings) if ratings else 6.5
        }

    def _extract_content_preferences(self, movies: List[Dict]) -> np.ndarray:
        """Create content-based preference embedding"""
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Combine movie overviews and titles
            texts = []
            for movie in movies:
                text = f"{movie['title']}. {movie.get('overview', '')}"
                texts.append(text)

            if texts:
                embeddings = model.encode(texts)
                # Return average embedding as user's content preference
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(384)  # Default MiniLM embedding size
        except Exception as e:
            print(f"Error in content preference extraction: {e}")
            return np.zeros(384)

    def _extract_temporal_preferences(self, movies: List[Dict]) -> Dict:
        """Extract user's temporal/era preferences"""
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
                'era_preference': np.mean(years),
                'era_variance': np.std(years),
                'modern_bias': sum(1 for y in years if y >= 2010) / len(years)
            }
        else:
            return {'era_preference': 2015, 'era_variance': 10, 'modern_bias': 0.7}

    def _get_default_preferences(self) -> Dict:
        """Default preferences for users with no valid history"""
        return {
            'genre_vector': np.ones(len(self.genre_mapping)) / len(self.genre_mapping),
            'rating_profile': {
                'preferred_rating_min': 6.0,
                'preferred_rating_avg': 7.0,
                'popularity_preference': 50.0,
                'quality_threshold': 6.5
            },
            'content_embedding': np.zeros(384),
            'temporal_pattern': {'era_preference': 2015, 'era_variance': 10, 'modern_bias': 0.7},
            'confidence_score': 0.1
        }


class FlexibleMovieEvaluator:
    """Evaluate any movie against user preferences without training dependency"""
    
    def __init__(self, feature_extractor: FlexibleFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.content_model = SentenceTransformer("all-MiniLM-L6-v2")

    def score_movie_for_user(self, movie: Dict, user_preferences: Dict) -> float:
        """Score any movie against user preferences"""
        try:
            # Extract movie features
            movie_features = self._extract_movie_features(movie)

            # Calculate similarity scores
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
                user_preferences['rating_profile']
            )

            temporal_score = self._calculate_temporal_match(
                movie_features['temporal_info'],
                user_preferences['temporal_pattern']
            )

            # Weighted combination
            final_score = (
                0.35 * genre_score +
                0.25 * content_score +
                0.25 * quality_score +
                0.15 * temporal_score
            )

            return final_score
        except Exception as e:
            print(f"Error scoring movie: {e}")
            return 0.0

    def _extract_movie_features(self, movie: Dict) -> Dict:
        """Extract features from any movie"""
        try:
            # Genre vector
            genre_vector = np.zeros(len(self.feature_extractor.genre_mapping))
            for genre in movie.get('genres', []):
                genre_name = genre['name'] if isinstance(genre, dict) else genre
                try:
                    genre_idx = list(self.feature_extractor.genre_mapping.values()).index(genre_name)
                    genre_vector[genre_idx] = 1.0
                except ValueError:
                    continue

            # Content embedding
            text = f"{movie['title']}. {movie.get('overview', '')}"
            content_embedding = self.content_model.encode([text])[0]

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
                    temporal_info['year'] = 2015
            else:
                temporal_info['year'] = 2015

            return {
                'genre_vector': genre_vector,
                'content_embedding': content_embedding,
                'quality_metrics': quality_metrics,
                'temporal_info': temporal_info
            }
        except Exception as e:
            print(f"Error extracting movie features: {e}")
            return self._get_default_movie_features()

    def _get_default_movie_features(self) -> Dict:
        """Return default movie features in case of error"""
        return {
            'genre_vector': np.zeros(len(self.feature_extractor.genre_mapping)),
            'content_embedding': np.zeros(384),
            'quality_metrics': {'rating': 0, 'popularity': 0, 'vote_count': 0},
            'temporal_info': {'year': 2015}
        }

    def _calculate_genre_similarity(self, movie_genres: np.ndarray, user_genres: np.ndarray) -> float:
        """Calculate genre similarity using cosine similarity"""
        if np.sum(user_genres) == 0 or np.sum(movie_genres) == 0:
            return 0.5  # Neutral score for no preferences
        
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
        """Calculate how well movie quality matches user preferences"""
        movie_rating = movie_quality['rating']
        user_min_rating = user_quality['preferred_rating_min']
        user_avg_rating = user_quality['preferred_rating_avg']

        # Penalty for movies below user's minimum threshold
        if movie_rating < user_min_rating:
            return 0.1

        # Score based on closeness to user's preferred range
        rating_diff = abs(movie_rating - user_avg_rating)
        rating_score = max(0, 1 - (rating_diff / 5.0))
        return rating_score

    def _calculate_temporal_match(self, movie_temporal: Dict, user_temporal: Dict) -> float:
        """Calculate temporal preference match"""
        movie_year = movie_temporal['year']
        user_era = user_temporal['era_preference']
        user_variance = user_temporal['era_variance']

        # Score based on how close the movie year is to user's preferred era
        year_diff = abs(movie_year - user_era)
        temporal_score = max(0, 1 - (year_diff / (user_variance * 2)))
        return temporal_score


class FlexibleRecommendationEngine:
    """Recommendation engine that works with any movie database"""
    
    def __init__(self, tmdb_api_key: str):
        self.feature_extractor = FlexibleFeatureExtractor(tmdb_api_key)
        self.movie_evaluator = FlexibleMovieEvaluator(self.feature_extractor)
        self.tmdb_api_key = tmdb_api_key

    def get_flexible_recommendations(self, user_history: List[int],
                                   mood: str = "neutral",
                                   count: int = 10) -> List[Dict]:
        """Get recommendations from any available movie pool"""
        try:
            # Extract user preferences from history
            user_preferences = self.feature_extractor.extract_user_preferences(user_history)

            # Get candidate movies (from multiple sources)
            candidate_movies = self._get_candidate_movies()

            # Score all candidates
            scored_movies = []
            for movie in candidate_movies:
                score = self.movie_evaluator.score_movie_for_user(movie, user_preferences)
                movie['similarity_score'] = score
                scored_movies.append(movie)

            # Sort by score and filter out already watched
            watched_ids = set(user_history)
            filtered_movies = [
                movie for movie in scored_movies
                if movie['id'] not in watched_ids
            ]

            # Sort by score and return top recommendations
            filtered_movies.sort(key=lambda x: x['similarity_score'], reverse=True)
            return filtered_movies[:count]
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def _get_candidate_movies(self) -> List[Dict]:
        """Get candidate movies from multiple TMDB endpoints"""
        candidates = []
        endpoints = [
            "https://api.themoviedb.org/3/movie/popular",
            "https://api.themoviedb.org/3/movie/top_rated",
            "https://api.themoviedb.org/3/discover/movie"
        ]

        for endpoint in endpoints:
            try:
                for page in range(1, 4):  # Get 3 pages per endpoint
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

        # Remove duplicates
        unique_movies = {}
        for movie in candidates:
            unique_movies[movie['id']] = movie
        
        return list(unique_movies.values())
