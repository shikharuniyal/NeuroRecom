#!/usr/bin/env python3
"""
Simple Movie Recommender Tester
Fetches random movies from TMDB and tests recommendations
"""

import json
import requests
import random
import time
from pathlib import Path
from datetime import datetime

class SimpleMovieTester:
    def __init__(self, tmdb_api_key: str):
        self.tmdb_api_key = tmdb_api_key
        self.api_url = "http://localhost:8000"
        self.data_dir = Path("data/users")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_random_movies(self, count=30):
        """Fetch random movies from TMDB"""
        print(f"🎬 Fetching {count} random movies from TMDB...")
        
        movies = []
        
        # Try different endpoints to get variety
        endpoints = [
            "https://api.themoviedb.org/3/movie/popular",
            "https://api.themoviedb.org/3/movie/top_rated",
            "https://api.themoviedb.org/3/discover/movie"
        ]
        
        for endpoint in endpoints:
            try:
                # Random page between 1-5 for variety
                page = random.randint(1, 5)
                
                params = {
                    "api_key": self.tmdb_api_key,
                    "page": page
                }
                
                response = requests.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                for movie in data.get("results", []):
                    movies.append({
                        "tmdb_id": movie["id"],
                        "title": movie["title"],
                        "genres": movie.get("genre_ids", []),
                        "rating": movie.get("vote_average", 0)
                    })
                    
                    if len(movies) >= count:
                        break
                        
            except Exception as e:
                print(f"❌ Error fetching from {endpoint}: {e}")
                continue
                
            if len(movies) >= count:
                break
        
        print(f"✅ Fetched {len(movies)} movies")
        return movies
    
    def create_test_user(self, movies, user_id=9999):
        """Create a test user with random movie history"""
        print(f"👤 Creating test user {user_id}...")
        
        # Randomly select 8-15 movies for history
        history_size = random.randint(8, 15)
        selected_movies = random.sample(movies, min(history_size, len(movies)))
        
        history = [movie["tmdb_id"] for movie in selected_movies]
        
        # Create some ratings for first few movies
        ratings = {}
        for i, movie_id in enumerate(history[:5]):
            rating = round(random.uniform(6.0, 9.5), 1)
            ratings[str(movie_id)] = rating
        
        # Random mood
        moods = ["happy", "neutral", "sad", "excited"]
        mood = random.choice(moods)
        
        user_data = {
            "user_id": user_id,
            "history": history,
            "ratings": ratings,
            "mood": mood,
            "favorite_genres": ["Action", "Comedy", "Drama"],
            "created_at": datetime.now().isoformat(),
            "test_info": {
                "watched_movies": [{"id": m["tmdb_id"], "title": m["title"]} 
                                 for m in selected_movies]
            }
        }
        
        # Save user file
        user_file = self.data_dir / f"{user_id}.json"
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=2)
        
        print(f"✅ Created test user with {len(history)} movies in history")
        print(f"📊 User mood: {mood}")
        print(f"🎬 Sample watched movies:")
        for movie in selected_movies[:3]:
            print(f"   - {movie['title']}")
        
        return user_data
    
    def get_recommendations(self, user_id, mood=None, count=10):
        """Get recommendations from the API"""
        print(f"\n🤖 Getting recommendations for user {user_id}...")
        
        params = {"count": count}
        if mood:
            params["mood"] = mood
        
        try:
            response = requests.get(
                f"{self.api_url}/recommend/{user_id}",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                
                print(f"✅ Got {len(recommendations)} recommendations")
                return recommendations
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def display_results(self, recommendations):
        """Display recommendations in a simple format"""
        if not recommendations:
            print("❌ No recommendations to display")
            return
        
        print(f"\n🎯 MOVIE RECOMMENDATIONS:")
        print("=" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Rating: {rec['rating']}/10")
            print(f"   Genres: {', '.join(rec['genres'])}")
            print(f"   Similarity Score: {rec.get('similarity_score', 'N/A')}")
            print(f"   Why recommended: {rec.get('recommendation_reason', 'Similar to your preferences')}")
            print()
    
    def run_test(self):
        """Run the complete test"""
        print("🚀 Starting Simple Movie Recommender Test")
        print("=" * 50)
        
        try:
            # Step 1: Fetch random movies
            movies = self.fetch_random_movies(30)
            if not movies:
                print("❌ Failed to fetch movies. Check your TMDB API key.")
                return
            
            # Step 2: Create test user
            user_data = self.create_test_user(movies)
            user_id = user_data["user_id"]
            
            # Step 3: Wait a moment for the system to process
            print("\n⏳ Waiting for system to be ready...")
            time.sleep(2)
            
            # Step 4: Get recommendations
            recommendations = self.get_recommendations(user_id, user_data["mood"])
            
            # Step 5: Display results
            self.display_results(recommendations)
            
            # Step 6: Test different moods
            print("\n🎭 Testing different moods...")
            for mood in ["happy", "sad"]:
                print(f"\n--- Mood: {mood.upper()} ---")
                mood_recs = self.get_recommendations(user_id, mood, 5)
                if mood_recs:
                    for i, rec in enumerate(mood_recs[:3], 1):
                        print(f"  {i}. {rec['title']} ({', '.join(rec['genres'])})")
            
            print("\n✅ Test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # IMPORTANT: Replace with your actual TMDB API key
    TMDB_API_KEY = "YOur_tmdb_api"
    
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("❌ Please replace 'YOUR_TMDB_API_KEY_HERE' with your actual TMDB API key")
        print("Get your API key from: https://www.themoviedb.org/settings/api")
    else:
        tester = SimpleMovieTester(TMDB_API_KEY)
        tester.run_test()
