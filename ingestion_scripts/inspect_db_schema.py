import sqlite3
import pandas as pd
import pathlib
import os

# --- Configuration ---
# This assumes the script is in 'ingestion_scripts' and 'data' is a sibling of 'src' at project_root
current_script_dir = pathlib.Path(__file__).resolve().parent
project_root = current_script_dir.parent 
DB_FILE_NAME = "perovskite_platform.sqlite"
DB_PATH = project_root / "data" / "processed" / DB_FILE_NAME

def inspect_all_table_schemas(db_path_str: str):
    """
    Connects to the SQLite database and prints the schema (columns, types)
    for all tables in the database.
    """
    conn = None
    print(f"--- Inspecting Database Schema for: {db_path_str} ---")
    
    if not os.path.exists(db_path_str):
        print(f"ERROR: Database file not found at '{db_path_str}'.")
        print("Please ensure '03_build_database.py' has been run successfully.")
        return

    try:
        conn = sqlite3.connect(db_path_str)
        print(f"Successfully connected to: {db_path_str}")
        cursor = conn.cursor()

        # Get a list of all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            return

        print(f"\nFound {len(tables)} tables: {[table[0] for table in tables]}")

        for table_tuple in tables:
            table_name = table_tuple[0]
            print(f"\n\n--- Schema for table: '{table_name}' ---")
            
            # PRAGMA table_info(table_name) is the SQL command to get schema info
            # It returns columns: cid, name, type, notnull, dflt_value, pk
            try:
                schema_df = pd.read_sql_query(f"PRAGMA table_info('{table_name}');", conn)
                if not schema_df.empty:
                    # Print relevant columns from PRAGMA table_info result
                    print(schema_df[['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']].to_string(index=False))
                else:
                    print("Could not retrieve schema information (PRAGMA table_info returned empty).")
            except pd.io.sql.DatabaseError as e: # Pandas specific SQL error
                print(f"  Error querying schema for table '{table_name}': {e}")
            except sqlite3.Error as e: # General SQLite error
                 print(f"  SQLite error querying schema for table '{table_name}': {e}")


    except sqlite3.Error as e:
        print(f"SQLite error during database inspection: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during database inspection: {e}")
    finally:
        if conn:
            conn.close()
            print("\n\n--- Database inspection complete. Connection closed. ---")

if __name__ == "__main__":
    inspect_all_table_schemas(str(DB_PATH)) # Convert Path object to string for sqlite3.connect