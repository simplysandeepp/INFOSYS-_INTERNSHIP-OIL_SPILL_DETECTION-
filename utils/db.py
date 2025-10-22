# ============================================================================
# DB.PY - Supabase Database Connection and Helper Functions
# ============================================================================

from supabase import create_client, Client
import os

# Initialize Supabase client
supabase: Client = None

# Try loading Streamlit secrets first (for Streamlit Cloud deployment)
try:
    import streamlit as st
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
    print("✅ Loaded Supabase credentials from Streamlit secrets")
except (ImportError, KeyError, FileNotFoundError) as e:
    print(f"⚠️ Could not load from Streamlit secrets: {e}")
    # Fallback to .env file (for local development)
    try:
        from dotenv import load_dotenv
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        load_dotenv(dotenv_path)
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        print(f"✅ Loaded Supabase credentials from .env file at {dotenv_path}")
    except Exception as env_error:
        print(f"⚠️ Could not load from .env: {env_error}")
        SUPABASE_URL = None
        SUPABASE_KEY = None

# Initialize Supabase client if credentials exist
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
        print(f"   URL: {SUPABASE_URL}")
    except Exception as e:
        supabase = None
        print(f"❌ Failed to initialize Supabase client: {e}")
else:
    supabase = None
    print("⚠️ Supabase credentials not found!")
    print("   Please set SUPABASE_URL and SUPABASE_KEY in:")
    print("   - .streamlit/secrets.toml (for Streamlit Cloud)")
    print("   - .env file (for local development)")


# -----------------------------
# Helper functions for database
# -----------------------------

DEFAULT_TABLE = "oil_detections"  # Default table name for oil spill detections


def insert_detection_data(data: dict, table_name: str = DEFAULT_TABLE):
    """
    Insert a detection record into the Supabase table.
    
    Args:
        data (dict): Detection data to insert with keys:
            - timestamp: ISO format timestamp
            - filename: Name of uploaded image
            - has_spill: Boolean indicating if spill was detected
            - coverage_percentage: Float percentage of image covered by spill
            - avg_confidence: Float average confidence score
            - max_confidence: Float maximum confidence score
            - detected_pixels: Integer number of pixels detected as spill
        table_name (str): Name of the Supabase table (default: oil_detections)
    
    Returns:
        Response object from Supabase
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📤 Inserting data into table '{table_name}':")
        print(f"   Data: {data}")
        
        response = supabase.table(table_name).insert(data).execute()
        
        print(f"✅ Successfully inserted record into '{table_name}'")
        print(f"   Response: {response}")
        
        return response
    
    except Exception as e:
        print(f"❌ Error inserting data into '{table_name}': {e}")
        raise


def fetch_all_detections(table_name: str = DEFAULT_TABLE):
    """
    Fetch all detection records from the Supabase table.
    
    Args:
        table_name (str): Name of the Supabase table (default: oil_detections)
    
    Returns:
        list: List of dictionaries containing detection records
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching all records from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").order('timestamp', desc=True).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} records from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching data from '{table_name}': {e}")
        raise


def fetch_recent_detections(table_name: str = DEFAULT_TABLE, limit: int = 10):
    """
    Fetch recent detection records from the Supabase table.
    
    Args:
        table_name (str): Name of the Supabase table (default: oil_detections)
        limit (int): Maximum number of records to fetch (default: 10)
    
    Returns:
        list: List of dictionaries containing recent detection records
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching {limit} recent records from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").order('timestamp', desc=True).limit(limit).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} recent records from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching recent data from '{table_name}': {e}")
        raise


def fetch_spill_detections_only(table_name: str = DEFAULT_TABLE):
    """
    Fetch only records where oil spills were detected.
    
    Args:
        table_name (str): Name of the Supabase table (default: oil_detections)
    
    Returns:
        list: List of dictionaries containing spill detection records
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📥 Fetching spill detections from table '{table_name}'...")
        
        response = supabase.table(table_name).select("*").eq('has_spill', True).order('timestamp', desc=True).execute()
        
        print(f"✅ Successfully fetched {len(response.data)} spill detections from '{table_name}'")
        
        return response.data
    
    except Exception as e:
        print(f"❌ Error fetching spill detections from '{table_name}': {e}")
        raise


def delete_detection(record_id: int, table_name: str = DEFAULT_TABLE):
    """
    Delete a specific detection record by ID.
    
    Args:
        record_id (int): ID of the record to delete
        table_name (str): Name of the Supabase table (default: oil_detections)
    
    Returns:
        Response object from Supabase
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"🗑️ Deleting record {record_id} from table '{table_name}'...")
        
        response = supabase.table(table_name).delete().eq('id', record_id).execute()
        
        print(f"✅ Successfully deleted record {record_id} from '{table_name}'")
        
        return response
    
    except Exception as e:
        print(f"❌ Error deleting record {record_id} from '{table_name}': {e}")
        raise


def get_database_stats(table_name: str = DEFAULT_TABLE):
    """
    Get statistics about the database records.
    
    Args:
        table_name (str): Name of the Supabase table (default: oil_detections)
    
    Returns:
        dict: Statistics including total records, spills detected, average coverage, etc.
    
    Raises:
        RuntimeError: If Supabase client is not initialized
    """
    if not supabase:
        raise RuntimeError("Supabase client not initialized. Check your credentials.")
    
    try:
        print(f"📊 Calculating statistics for table '{table_name}'...")
        
        # Fetch all records
        all_records = fetch_all_detections(table_name)
        
        if not all_records:
            return {
                'total_records': 0,
                'spills_detected': 0,
                'clean_images': 0,
                'avg_coverage': 0.0,
                'avg_confidence': 0.0,
                'detection_rate': 0.0
            }
        
        # Calculate statistics
        total_records = len(all_records)
        spills_detected = sum(1 for r in all_records if r.get('has_spill', False))
        clean_images = total_records - spills_detected
        
        # Calculate averages
        avg_coverage = sum(r.get('coverage_percentage', 0) for r in all_records) / total_records
        avg_confidence = sum(r.get('avg_confidence', 0) for r in all_records) / total_records
        detection_rate = (spills_detected / total_records * 100) if total_records > 0 else 0.0
        
        stats = {
            'total_records': total_records,
            'spills_detected': spills_detected,
            'clean_images': clean_images,
            'avg_coverage': round(avg_coverage, 2),
            'avg_confidence': round(avg_confidence, 3),
            'detection_rate': round(detection_rate, 2)
        }
        
        print(f"✅ Statistics calculated: {stats}")
        
        return stats
    
    except Exception as e:
        print(f"❌ Error calculating statistics for '{table_name}': {e}")
        raise


def test_connection():
    """
    Test the Supabase database connection.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        print("🔍 Testing Supabase connection...")
        
        # Try to fetch one record to test connection
        response = supabase.table(DEFAULT_TABLE).select("*").limit(1).execute()
        
        print("✅ Supabase connection successful!")
        print("   Table '" + DEFAULT_TABLE + "' is accessible")
        
        return True
    
    except Exception as e:
        print("❌ Supabase connection test failed: " + str(e))
        return False


# Test connection when module is imported
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SUPABASE DATABASE CONNECTION TEST")
    print("="*60 + "\n")
    
    if test_connection():
        print("\n✅ All database tests passed!")
        
        # Try to get stats
        try:
            stats = get_database_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                print("   " + str(key) + ": " + str(value))
        except Exception as e:
            print("\n⚠️ Could not fetch statistics: " + str(e))
    else:
        print("\n❌ Database connection test failed!")
        print("\nTroubleshooting:")
        print("1. Check if SUPABASE_URL and SUPABASE_KEY are set correctly")
        print("2. Verify your Supabase project is active")
        print("3. Ensure the 'oil_detections' table exists in your database")
        print("4. Check if your API key has the correct permissions")