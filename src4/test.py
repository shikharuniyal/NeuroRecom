import psycopg2
import json
from datetime import datetime
import sys
import os
from typing import Dict, List, Optional

# Import your recommendation system
from main import HybridRecommendationAPI

class MovieRecommendationTester:
    """Test class for the hybrid movie recommendation system with database integration"""
    
    def __init__(self, connection_params: Dict, tmdb_api_key: str):
        self.connection_params = connection_params
        self.tmdb_api_key = tmdb_api_key
        self.api = HybridRecommendationAPI(tmdb_api_key)
        
    def connect_to_database(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return None
    
    def fetch_user_data(self, user_id: int) -> Optional[Dict]:
        """Fetch specific user data for recommendations"""
        conn = self.connect_to_database()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            
            # Use the database function to get formatted user data
            cursor.execute("SELECT get_user_recommendation_data(%s);", (user_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                user_data = json.loads(result[0])
                print(f"✅ Fetched data for user {user_id}: {user_data['username']}")
                return user_data
            else:
                print(f"❌ No data found for user {user_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching user data: {e}")
            return None
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def fetch_all_users_data(self) -> List[Dict]:
        """Fetch all users data for collaborative filtering"""
        conn = self.connect_to_database()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            
            # Use the database function to get all users data
            cursor.execute("SELECT get_all_users_recommendation_data();")
            result = cursor.fetchone()
            
            if result and result[0]:
                all_users_data = json.loads(result[0])
                print(f"✅ Fetched data for {len(all_users_data)} users")
                return all_users_data
            else:
                print("❌ No users data found")
                return []
                
        except Exception as e:
            print(f"❌ Error fetching all users data: {e}")
            return []
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def save_recommendations(self, user_id: int, recommendations: List[Dict], 
                           session_info: Dict) -> bool:
        """Save recommendations to the database"""
        conn = self.connect_to_database()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            # Clear existing recommendations for this user
            cursor.execute(
                "DELETE FROM user_recommendations WHERE user_id = %s AND is_active = true",
                (user_id,)
            )
            
            # Insert new recommendations
            insert_query = """
            INSERT INTO user_recommendations (
                user_id, tmdb_id, title, vote_average, release_date, popularity, 
                similarity_score, genres, overview, recommendation_method, 
                recommendation_reason, poster_path
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for rec in recommendations:
                # Parse release_date
                release_date = None
                if rec.get('release_date'):
                    try:
                        release_date = datetime.strptime(rec['release_date'], '%Y-%m-%d').date()
                    except:
                        release_date = None
                
                # Create recommendation reason
                reason = f"Recommended via {rec.get('recommendation_method', 'hybrid')} "
                reason += f"with {rec.get('similarity_score', 0):.3f} similarity score"
                
                cursor.execute(insert_query, (
                    user_id,
                    rec['tmdb_id'],
                    rec['title'],
                    rec.get('vote_average', 0.0),
                    release_date,
                    rec.get('popularity', 0.0),
                    rec.get('similarity_score', 0.0),
                    rec.get('genres', []),
                    rec.get('overview', ''),
                    rec.get('recommendation_method', 'hybrid'),
                    reason,
                    rec.get('poster_path', '')
                ))
            
            # Save session information
            cursor.execute("""
            INSERT INTO recommendation_sessions (
                user_id, session_mood, time_watched, recommendation_count, 
                method_used, cf_readiness_data, processing_time_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                session_info.get('mood', 'neutral'),
                session_info.get('time_watched', 'day'),
                len(recommendations),
                session_info.get('method_used', 'hybrid'),
                json.dumps(session_info.get('cf_readiness', {})),
                session_info.get('processing_time_ms', 0)
            ))
            
            conn.commit()
            print(f"✅ Saved {len(recommendations)} recommendations for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving recommendations: {e}")
            conn.rollback()
            return False
        finally:
            if conn:
                cursor.close()
                conn.close()
    
    def test_recommendation_for_user(self, user_id: int, mood: str = "neutral", 
                                   time_watched: str = "day", count: int = 10) -> bool:
        """Test the complete recommendation pipeline for a specific user"""
        
        print(f"\n🎬 Testing recommendations for user {user_id}")
        print("=" * 50)
        
        # Record start time
        start_time = datetime.now()
        
        # Step 1: Fetch user data
        print("📊 Step 1: Fetching user data...")
        user_data = self.fetch_user_data(user_id)
        if not user_data:
            return False
        
        # Update user preferences if provided
        user_data.update({
            'mood': mood,
            'time_watched': time_watched,
            'count': count
        })
        
        # Step 2: Fetch all users data
        print("👥 Step 2: Fetching all users data for collaborative filtering...")
        all_users_data = self.fetch_all_users_data()
        if not all_users_data:
            print("⚠️ Warning: No collaborative filtering data available")
        
        # Step 3: Generate recommendations
        print("🤖 Step 3: Generating recommendations...")
        try:
            recommendations = self.api.recommend(user_data, all_users_data)
            
            if not recommendations:
                print("❌ No recommendations generated")
                return False
                
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            print(f"✅ Generated {len(recommendations)} recommendations")
            
            # Display sample recommendations
            print("\n🎯 Sample Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec['title']} ({rec.get('similarity_score', 0):.3f} score)")
                print(f"   Method: {rec.get('recommendation_method', 'unknown')}")
                print(f"   Genres: {', '.join(rec.get('genres', []))}")
                
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return False
        
        # Step 4: Save recommendations
        print("\n💾 Step 4: Saving recommendations to database...")
        session_info = {
            'mood': mood,
            'time_watched': time_watched,
            'method_used': recommendations[0].get('recommendation_method', 'hybrid') if recommendations else 'unknown',
            'cf_readiness': {'total_users': len(all_users_data)},
            'processing_time_ms': int(processing_time)
        }
        
        success = self.save_recommendations(user_id, recommendations, session_info)
        
        if success:
            print(f"🎉 Test completed successfully in {processing_time:.2f}ms!")
            return True
        else:
            return False
    
    def test_all_users(self) -> Dict:
        """Test recommendations for all users in the database"""
        
        print("\n🧪 Testing recommendations for all users")
        print("=" * 50)
        
        conn = self.connect_to_database()
        if not conn:
            return {"success": False, "error": "Database connection failed"}
        
        results = {"success": 0, "failed": 0, "errors": []}
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM users ORDER BY id")
            users = cursor.fetchall()
            
            print(f"Found {len(users)} users to test")
            
            for user_id, username in users:
                print(f"\n👤 Testing user: {username} (ID: {user_id})")
                try:
                    success = self.test_recommendation_for_user(user_id)
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"User {user_id}: Recommendation failed")
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"User {user_id}: {str(e)}")
                    
        except Exception as e:
            results["errors"].append(f"Database query failed: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                conn.close()
        
        return results

def main():
    """Main function to run the tests"""
    
    # Database connection parameters - UPDATE THESE WITH YOUR ACTUAL CREDENTIALS
    connection_params = {
        'database': 'movie_recommendation_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': '5432'
    }
    
    # TMDB API key - UPDATE WITH YOUR ACTUAL API KEY
    tmdb_api_key = 'your_tmdb_api_key_here'
    
    # Initialize the tester
    tester = MovieRecommendationTester(connection_params, tmdb_api_key)
    
    print("🎬 Movie Recommendation System Tester")
    print("=" * 40)
    
    # Test options
    test_choice = input("""
Choose test option:
1. Test specific user (enter user ID)
2. Test all users
3. Custom test with mood/time preferences

Enter choice (1-3): """).strip()
    
    if test_choice == "1":
        try:
            user_id = int(input("Enter user ID: "))
            tester.test_recommendation_for_user(user_id)
        except ValueError:
            print("❌ Invalid user ID")
            
    elif test_choice == "2":
        results = tester.test_all_users()
        print(f"\n📊 Final Results:")
        print(f"✅ Successful: {results['success']}")
        print(f"❌ Failed: {results['failed']}")
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
                
    elif test_choice == "3":
        try:
            user_id = int(input("Enter user ID: "))
            mood = input("Enter mood (happy/sad/excited/neutral): ").strip() or "neutral"
            time_watched = input("Enter time preference (day/night): ").strip() or "day"
            count = int(input("Enter number of recommendations (default 10): ") or "10")
            
            tester.test_recommendation_for_user(user_id, mood, time_watched, count)
        except ValueError:
            print("❌ Invalid input")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
