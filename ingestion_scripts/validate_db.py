# ingestion_scripts/04_validate_db.py
import sqlite3
import pandas as pd
import pathlib
import os # Added for os.path.join for consistency

def validate_database_contents():
    print("--- Starting Script 04: Validate Database Contents ---")
    
    current_script_dir = pathlib.Path(__file__).resolve().parent
    project_root = current_script_dir.parent
    db_file_name = "perovskite_platform.sqlite"
    db_path = os.path.join(project_root, "data", "processed", db_file_name) # Use os.path.join

    if not os.path.exists(db_path): # Use os.path.exists for string paths
        print(f"ERROR: Database file not found at {db_path}. Cannot validate.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print(f"Successfully connected to SQLite database: {db_path}")
        cursor = conn.cursor()

        # 1. Check if all expected tables exist
        print("\n1. Checking for expected tables...")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables_in_db = [row[0] for row in cursor.fetchall()]
        expected_tables = sorted([
            "Materials", "BandStructureData", "KLabelsData", 
            "HighSymmetryPointsData", "OpticalSpectraData", "DensityOfStatesData"
        ])
        
        all_tables_exist = True
        for table_name in expected_tables:
            if table_name in tables_in_db:
                print(f"  OK: Table '{table_name}' exists.")
            else:
                print(f"  ERROR: Expected table '{table_name}' NOT FOUND.")
                all_tables_exist = False
        
        if not all_tables_exist:
            print("Critical error: Not all expected tables exist. Halting further validation.")
            return

        # 2. Check schema for Materials table (example)
        print("\n2. Checking 'Materials' table schema (name, type, notnull, pk)...")
        materials_schema_df = pd.read_sql_query("PRAGMA table_info(Materials);", conn)
        print(materials_schema_df[['name', 'type', 'notnull', 'pk']].to_string())
        # You would ideally compare this against your expected schema programmatically

        # 3. Check row counts
        print("\n3. Checking row counts...")
        for table_name in expected_tables:
            count_df = pd.read_sql_query(f"SELECT COUNT(*) as row_count FROM {table_name};", conn)
            print(f"  - Rows in '{table_name}': {count_df['row_count'].iloc[0]}")
            if table_name == "Materials" and count_df['row_count'].iloc[0] != 7: # Assuming 7 materials
                print(f"    WARNING: Expected 7 rows in Materials, found {count_df['row_count'].iloc[0]}")

        # 4. Check sample data from Materials table
        print("\n4. Sample data from 'Materials' for x_value = 0 and x_value = 6:")
        sample_materials_df = pd.read_sql_query(
            "SELECT * FROM Materials WHERE material_x_value = 0 OR material_x_value = 6;", 
            conn
        )
        print(sample_materials_df.to_string())
        if len(sample_materials_df) != 2:
             print("    WARNING: Did not retrieve exactly 2 rows for x=0 and x=6 in Materials table.")


        # 5. Check for data in spectral tables (example for OpticalSpectraData)
        print("\n5. Check for data integrity in 'OpticalSpectraData' for x_value=2 and ABSORPTION type...")
        sample_optical_df = pd.read_sql_query(
            "SELECT COUNT(*) as count, MIN(photon_energy_ev) as min_e, MAX(photon_energy_ev) as max_e "
            "FROM OpticalSpectraData WHERE material_x_value = 2 AND spectrum_type = 'ABSORPTION';",
            conn
        )
        if not sample_optical_df.empty and sample_optical_df['count'].iloc[0] > 0:
            print(f"  OK: Found {sample_optical_df['count'].iloc[0]} absorption data points for x=2. Energy range: {sample_optical_df['min_e'].iloc[0]:.2f} - {sample_optical_df['max_e'].iloc[0]:.2f} eV.")
        else:
            print("  WARNING: No absorption data found for x=2, or query failed.")

        print("\n--- Database Validation Checks Complete ---")

    except sqlite3.Error as e:
        print(f"SQLite error during validation: {e}")
    except pd.io.sql.DatabaseError as e:
        print(f"Pandas SQL query error during validation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during validation: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed after validation.")

if __name__ == "__main__":
    validate_database_contents()