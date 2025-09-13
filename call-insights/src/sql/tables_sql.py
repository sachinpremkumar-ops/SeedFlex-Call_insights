import psycopg2
from psycopg2.extras import RealDictCursor
import json
from src.utils.s3_utils import get_secret

def connect_to_rds():
    secret = get_secret()
    try:
        conn = psycopg2.connect(
            host=secret["host"],
            user=secret["username"],
            password=secret["password"],
            dbname=secret["database"],
            port=int(secret["port"]),
            cursor_factory=RealDictCursor
        )
        print("✅ Connection successful")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return None


def create_tables(connection):
    with connection.cursor() as cursor:
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    workflow_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_size BIGINT,
                    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    transcript_id SERIAL PRIMARY KEY,
                    workflow_id TEXT REFERENCES calls(workflow_id) ON DELETE CASCADE,
                    transcript_text TEXT ,
                    translated_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id SERIAL PRIMARY KEY,
                    workflow_id TEXT REFERENCES calls(workflow_id) ON DELETE CASCADE,
                    topic TEXT,
                    abstract_summary TEXT,
                    key_points TEXT,
                    action_items TEXT,
                    sentiment_label TEXT,
                    sentiment_scores TEXT,
                    embeddings vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    execution_id SERIAL PRIMARY KEY,
                    workflow_id TEXT,
                    agent_name TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP, 
                    execution_time_ms INT,
                    status TEXT,
                    error_message TEXT
                );
            """)
            connection.commit()
            print("✅ Tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            connection.rollback()

def insert_data_calls(workflow_id,file_key,file_size,uploaded_at):
    try:
        connection=connect_to_rds()
        with connection.cursor() as cursor:
            create_tables(connection)
            cursor.execute("""
                INSERT INTO calls(workflow_id,file_name,file_size,uploaded_at)
                VALUES (%s,%s,%s,%s)
            """, (workflow_id, file_key, file_size, uploaded_at))
            connection.commit()
            print("Data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")
        if 'connection' in locals():
            connection.rollback()

def get_workflow_id(file_key):
    try:
        connection=connect_to_rds()
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT workflow_id FROM calls WHERE file_key = %s
            """, (file_key,))
            return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error getting workflow id: {e}")
        return None

def insert_data_transcripts(workflow_id,file_key,transcript_text,translated_text,created_at):
    try:
        connection=connect_to_rds()
        
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO transcripts(workflow_id,file_key,transcript_text,translated_text,created_at)
                VALUES (%s,%s,%s,%s)
            """, (get_workflow_id(file_key),transcript_text,translated_text,created_at))
            connection.commit()
            print("Data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")
        if 'connection' in locals():
            connection.rollback()

def insert_data_analyses(workflow_id,file_key,topic,abstract_summary,key_points,action_items,sentiment_label,sentiment_scores,embeddings):
    try:
        connection=connect_to_rds()
        with connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO analyses(workflow_id,file_key,topic,abstract_summary,key_points,action_items,sentiment_label,sentiment_scores,embeddings)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (get_workflow_id(file_key),topic,abstract_summary,key_points,action_items,sentiment_label,sentiment_scores,embeddings))
            connection.commit()
            print("Data inserted successfully!")
    except Exception as e:
        print(f"Error inserting data: {e}")
        if 'connection' in locals():
            connection.rollback()

def insert_data_all(workflow_id,file_key, file_size, uploaded_at,
                    transcription=None, translation=None,
                    topic=None, summary=None, key_points=None,
                    action_items=None, sentiment_label=None,
                    sentiment_scores=None, embeddings=None):
    """
    Insert call data, transcripts, and analysis into the database.
    embeddings should be a list or tuple of 1536 floats.
    """

    try:
        connection = connect_to_rds()  # Make sure this returns a psycopg2 connection
        with connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:

            # Insert into calls and get call_id
            create_tables(connection)
            print("Tables created successfully!")


            cursor.execute("""
                INSERT INTO calls(workflow_id,file_key, file_size, uploaded_at)
                VALUES (%s, %s, %s)
                RETURNING workflow_id
            """, (workflow_id,file_key, file_size, uploaded_at))

            result = cursor.fetchone()
            if not result:
                print("ERROR: No workflow_id returned from INSERT")
                connection.rollback()
                return

            workflow_id = result['workflow_id']

            # Insert transcript if available
            if transcription or translation:
                cursor.execute("""
                    INSERT INTO transcripts(workflow_id, transcript_text, translated_text, created_at)
                    VALUES (%s, %s, %s, NOW())
                """, (workflow_id, transcription, translation))

            # Insert analysis if any field is provided
            if any([topic, summary, key_points, action_items, sentiment_label, sentiment_scores, embeddings]):
                # Ensure embeddings is a list of floats or None
                if embeddings:
                    if not isinstance(embeddings, (list, tuple)):
                        raise ValueError("Embeddings must be a list or tuple of floats")
                    if len(embeddings) != 1536:
                        raise ValueError("Embeddings must have length 1536")

                cursor.execute("""
                    INSERT INTO analyses(
                        workflow_id, topic, abstract_summary, key_points, action_items,
                        sentiment_label, sentiment_scores, embeddings
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """, (workflow_id, topic, summary, key_points, action_items, sentiment_label, sentiment_scores, embeddings))

            # Commit once at the end
            connection.commit()
            print("All data inserted successfully!")

    except Exception as e:
        print(f"Error inserting data: {e}")
        if 'connection' in locals():
            connection.rollback()
    finally:
        if 'connection' in locals():
            connection.close()
