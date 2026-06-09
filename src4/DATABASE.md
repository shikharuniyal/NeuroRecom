


-- Create the database
CREATE DATABASE movie_recommendation_db;

-- Connect to the database
\c movie_recommendation_db;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Users table (based on your existing structure)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    display_name VARCHAR(150),
    avatar_url VARCHAR(500),
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Watched movies table (enhanced from your existing structure)
CREATE TABLE watched_movies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id INTEGER, -- Your internal movie ID
    tmdb_id INTEGER NOT NULL, -- TMDB API ID
    title VARCHAR(255) NOT NULL,
    watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rating DECIMAL(3,1) CHECK (rating >= 0 AND rating <= 10),
    current_mood VARCHAR(50),
    genres TEXT[],
    vote_average DECIMAL(3,1),
    popularity DECIMAL(8,3),
    overview TEXT,
    poster_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Mood selections table (from your existing structure)
CREATE TABLE mood_selections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    mood VARCHAR(50) NOT NULL,
    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    page VARCHAR(100)
);

-- 4. User preferences table (for recommendation system)
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    favorite_genres TEXT[] DEFAULT '{}',
    disliked_genres TEXT[] DEFAULT '{}',
    min_rating DECIMAL(3,1) DEFAULT 6.0,
    preferred_era_start INTEGER,
    preferred_era_end INTEGER,
    watch_time_preference VARCHAR(20) DEFAULT 'any',
    mood_preference VARCHAR(20) DEFAULT 'neutral',
    language_preference VARCHAR(10) DEFAULT 'en',
    adult_content BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id)
);

-- 5. User recommendations table
CREATE TABLE user_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id INTEGER NOT NULL,
    title VARCHAR(255) NOT NULL,
    vote_average DECIMAL(3,1),
    release_date DATE,
    popularity DECIMAL(8,3),
    similarity_score DECIMAL(5,4) NOT NULL,
    genres TEXT[],
    overview TEXT,
    poster_path VARCHAR(255),
    recommendation_method VARCHAR(50) DEFAULT 'hybrid',
    recommendation_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '24 hours'),
    is_active BOOLEAN DEFAULT true,
    UNIQUE(user_id, tmdb_id)
);

-- 6. Recommendation sessions table (for tracking)
CREATE TABLE recommendation_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_mood VARCHAR(50),
    time_watched VARCHAR(20),
    recommendation_count INTEGER DEFAULT 0,
    method_used VARCHAR(50),
    cf_readiness_data JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. User interactions table (for feedback)
CREATE TABLE user_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tmdb_id INTEGER NOT NULL,
    interaction_type VARCHAR(50) NOT NULL,
    recommendation_id INTEGER REFERENCES user_recommendations(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_watched_movies_user_id ON watched_movies(user_id);
CREATE INDEX idx_watched_movies_tmdb_id ON watched_movies(tmdb_id);
CREATE INDEX idx_watched_movies_user_rating ON watched_movies(user_id, rating) WHERE rating IS NOT NULL;
CREATE INDEX idx_user_recommendations_user_id ON user_recommendations(user_id);
CREATE INDEX idx_user_recommendations_active ON user_recommendations(user_id, is_active, expires_at);
CREATE INDEX idx_user_recommendations_score ON user_recommendations(user_id, similarity_score DESC);
CREATE INDEX idx_mood_selections_user_id ON mood_selections(user_id);
CREATE INDEX idx_user_interactions_user_id ON user_interactions(user_id);

-- Create update timestamp triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_watched_movies_updated_at BEFORE UPDATE ON watched_movies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for easy data access
CREATE OR REPLACE VIEW user_profile_view AS
SELECT 
    u.id as user_id,
    u.username,
    u.email,
    u.display_name,
    COALESCE(up.favorite_genres, '{}') as favorite_genres,
    COALESCE(up.mood_preference, 'neutral') as mood,
    COALESCE(up.watch_time_preference, 'day') as time_watched,
    (SELECT COUNT(*) FROM watched_movies wm WHERE wm.user_id = u.id) as total_movies_watched,
    (SELECT AVG(rating) FROM watched_movies wm WHERE wm.user_id = u.id AND rating IS NOT NULL) as avg_rating,
    (SELECT COUNT(*) FROM watched_movies wm WHERE wm.user_id = u.id AND rating IS NOT NULL) as total_ratings_given,
    u.created_at,
    u.updated_at
FROM users u
LEFT JOIN user_preferences up ON u.id = up.user_id;

-- Function to get user data formatted for recommendation model
CREATE OR REPLACE FUNCTION get_user_recommendation_data(target_user_id INTEGER)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'user_id', u.id,
        'username', u.username,
        'user_history', COALESCE(
            (SELECT array_agg(wm.tmdb_id ORDER BY wm.watched_at DESC) 
             FROM watched_movies wm WHERE wm.user_id = u.id), 
            ARRAY[]::INTEGER[]
        ),
        'ratings', COALESCE(
            (SELECT jsonb_object_agg(wm.tmdb_id::text, wm.rating) 
             FROM watched_movies wm WHERE wm.user_id = u.id AND wm.rating IS NOT NULL),
            '{}'::JSONB
        ),
        'favourite_genres', COALESCE(up.favorite_genres, ARRAY[]::TEXT[]),
        'mood', COALESCE(up.mood_preference, 'neutral'),
        'time_watched', COALESCE(up.watch_time_preference, 'day'),
        'count', 10
    )
    INTO result
    FROM users u
    LEFT JOIN user_preferences up ON u.id = up.user_id
    WHERE u.id = target_user_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Function to get all users data for collaborative filtering
CREATE OR REPLACE FUNCTION get_all_users_recommendation_data()
RETURNS JSON AS $$
BEGIN
    RETURN (
        SELECT json_agg(
            json_build_object(
                'user_id', u.id,
                'username', u.username,
                'user_history', COALESCE(user_movies.movie_list, ARRAY[]::INTEGER[]),
                'ratings', COALESCE(user_ratings.rating_map, '{}'::JSONB),
                'favourite_genres', COALESCE(up.favorite_genres, ARRAY[]::TEXT[]),
                'mood', COALESCE(up.mood_preference, 'neutral'),
                'time_watched', COALESCE(up.watch_time_preference, 'day')
            )
        )
        FROM users u
        LEFT JOIN user_preferences up ON u.id = up.user_id
        LEFT JOIN (
            SELECT 
                user_id, 
                array_agg(tmdb_id ORDER BY watched_at DESC) as movie_list
            FROM watched_movies 
            GROUP BY user_id
        ) user_movies ON u.id = user_movies.user_id
        LEFT JOIN (
            SELECT 
                user_id,
                jsonb_object_agg(tmdb_id::text, rating) as rating_map
            FROM watched_movies 
            WHERE rating IS NOT NULL
            GROUP BY user_id
        ) user_ratings ON u.id = user_ratings.user_id
        WHERE user_movies.movie_list IS NOT NULL
    );
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO users (username, email, display_name) VALUES
('john_doe', 'john@example.com', 'John Doe'),
('jane_smith', 'jane@example.com', 'Jane Smith'),
('mike_wilson', 'mike@example.com', 'Mike Wilson'),
('sarah_davis', 'sarah@example.com', 'Sarah Davis');

-- Sample user preferences
INSERT INTO user_preferences (user_id, favorite_genres, mood_preference, watch_time_preference) VALUES
(1, ARRAY['Action', 'Thriller', 'Sci-Fi'], 'excited', 'night'),
(2, ARRAY['Romance', 'Drama', 'Comedy'], 'happy', 'day'),
(3, ARRAY['Horror', 'Mystery', 'Thriller'], 'neutral', 'night'),
(4, ARRAY['Family', 'Animation', 'Adventure'], 'happy', 'day');

-- Sample watched movies (using real TMDB IDs)
INSERT INTO watched_movies (user_id, tmdb_id, title, rating, genres, vote_average) VALUES
(1, 550, 'Fight Club', 9.0, ARRAY['Drama', 'Thriller'], 8.4),
(1, 13, 'Forrest Gump', 8.5, ARRAY['Drama', 'Romance'], 8.5),
(1, 680, 'Pulp Fiction', 9.5, ARRAY['Crime', 'Drama'], 8.9),
(2, 19404, 'Dilwale Dulhania Le Jayenge', 8.0, ARRAY['Romance', 'Drama'], 8.7),
(2, 597, 'Titanic', 7.5, ARRAY['Romance', 'Drama'], 7.9),
(3, 11, 'Star Wars', 8.8, ARRAY['Adventure', 'Action', 'Science Fiction'], 8.6),
(3, 157336, 'Interstellar', 9.2, ARRAY['Adventure', 'Drama', 'Science Fiction'], 8.4),
(4, 862, 'Toy Story', 8.0, ARRAY['Animation', 'Comedy', 'Family'], 8.0),
(4, 12, 'Finding Nemo', 8.5, ARRAY['Animation', 'Family'], 8.0);
