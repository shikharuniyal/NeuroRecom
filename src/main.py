#!/usr/bin/env python3
"""
Enhanced Hybrid Movie Recommender System
Combines Content-Based + Collaborative Filtering + Context + Neural Networks
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# Web framework
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import the flexible recommendation engine
from recom import FlexibleRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TMDB API Key
TMDB_API_KEY = "YOur_tmdb_api"

class MovieRating(BaseModel):
    """Pydantic model for movie ratings"""
    user_id: int
    movie_id: int
    rating: float = 0.0

class UserPreferences(BaseModel):
    """User preference model"""
    user_id: int
    favorite_genres: List[str] = []
    mood: str = "neutral"
    watch_time_preference: str = "any"

class SimpleRecommendationSystem:
    """Simplified recommendation system using the flexible engine"""
    
    def __init__(self, tmdb_api_key: str, data_dir: str = "data"):
        self.tmdb_api_key = tmdb_api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "users").mkdir(exist_ok=True)
        
        # Initialize the flexible recommendation engine
        self.flexible_engine = FlexibleRecommendationEngine(tmdb_api_key)

    def create_sample_user(self, user_id: int, movie_history: List[int] = None) -> Dict:
        """Create a sample user with given history"""
        if movie_history is None:
            movie_history = [550, 680, 13, 278, 238]  # Default popular movies
        
        user_data = {
            "user_id": user_id,
            "history": movie_history,
            "ratings": {},
            "mood": "neutral",
            "favorite_genres": ["Action", "Drama"],
            "created_at": datetime.now().isoformat()
        }
        
        # Save user data
        user_file = self.data_dir / "users" / f"{user_id}.json"
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=2)
        
        return user_data

    def get_user_data(self, user_id: int) -> Dict:
        """Get user data from file"""
        user_file = self.data_dir / "users" / f"{user_id}.json"
        if user_file.exists():
            with open(user_file) as f:
                return json.load(f)
        else:
            # Create a sample user if doesn't exist
            return self.create_sample_user(user_id)

    def get_recommendations(self, user_id: int, mood: str = "neutral", count: int = 10) -> List[Dict]:
        """Get recommendations for a user"""
        try:
            user_data = self.get_user_data(user_id)
            recommendations = self.flexible_engine.get_flexible_recommendations(
                user_data["history"], mood, count
            )
            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            return []

# Initialize the recommendation system
recommender = SimpleRecommendationSystem(TMDB_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup code
    logger.info("Starting up Movie Recommender...")
    try:
        logger.info("Recommender system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
    
    yield  # App is running
    
    # Shutdown code
    logger.info("Shutting down Movie Recommender...")

app = FastAPI(
    title="Movie Recommender System",
    version="1.0.0",
    description="AI-powered movie recommendations",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Movie Recommendation System API", "status": "running"}

@app.get("/recommend/{user_id}")
async def get_recommendations(
    user_id: int,
    mood: str = "neutral",
    count: int = 10
):
    """Get movie recommendations for a user"""
    try:
        recommendations = recommender.get_recommendations(user_id, mood, count)
        
        # Format response
        formatted_recs = []
        for rec in recommendations:
            formatted_recs.append({
                "tmdb_id": rec["id"],
                "title": rec["title"],
                "overview": rec.get("overview", "")[:200],
                "rating": rec.get("vote_average", 0),
                "genres": [g["name"] if isinstance(g, dict) else g for g in rec.get("genres", [])],
                "similarity_score": rec.get("similarity_score", 0),
                "poster_path": rec.get("poster_path", "")
            })

        return {
            "user_id": user_id,
            "mood": mood,
            "recommendations": formatted_recs,
            "count": len(formatted_recs)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.post("/user/{user_id}/rate")
async def rate_movie(user_id: int, rating: MovieRating):
    """Add or update a movie rating for a user"""
    try:
        user_data = recommender.get_user_data(user_id)
        
        # Add rating
        user_data["ratings"][str(rating.movie_id)] = rating.rating
        
        # Add to history if not already there
        if rating.movie_id not in user_data["history"]:
            user_data["history"].append(rating.movie_id)

        # Save updated user data
        user_file = recommender.data_dir / "users" / f"{user_id}.json"
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=2)

        return {"message": "Rating added successfully", "rating": rating.rating}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rating error: {str(e)}")

@app.get("/user/{user_id}")
async def get_user_profile(user_id: int):
    """Get user profile and analytics"""
    try:
        user_data = recommender.get_user_data(user_id)
        ratings = user_data.get("ratings", {})
        
        analytics = {
            "user_id": user_id,
            "total_movies_watched": len(user_data["history"]),
            "total_ratings_given": len(ratings),
            "average_rating": np.mean(list(ratings.values())) if ratings else 0,
            "favorite_genres": user_data.get("favorite_genres", []),
            "recent_history": user_data["history"][-5:] if user_data["history"] else []
        }
        
        return analytics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile error: {str(e)}")

@app.post("/user/{user_id}/history")
async def update_user_history(user_id: int, movie_ids: List[int]):
    """Update user history for testing purposes"""
    try:
        user_data = recommender.get_user_data(user_id)
        user_data["history"] = movie_ids
        user_data["updated_at"] = datetime.now().isoformat()

        # Save updated data
        user_file = recommender.data_dir / "users" / f"{user_id}.json"
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=2)

        return {"message": "History updated", "new_history": movie_ids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
