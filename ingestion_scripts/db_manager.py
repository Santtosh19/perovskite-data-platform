# src/db_manager.py
import sqlite3  # Python's built-in library for SQLite
import pandas as pd # For loading DataFrames

def create_connection(db_file_path_str: str):
    """ 
    Create a database connection to a SQLite database.
    If the database file does not exist, it will be created.
    
    Args:
        db_file_path_str (str): The path to the SQLite database file.
    
    Returns:
        sqlite3.Connection object or None if connection failed.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file_path_str)
        print(f"Successfully connected to SQLite database: {db_file_path_str}")
    except sqlite3.Error as e:
        print(f"Error connecting to database {db_file_path_str}: {e}")
    return conn

def execute_sql_script(conn: sqlite3.Connection, sql_script_string: str):
    """ 
    Execute multiple SQL statements from a single string (e.g., for schema creation).
    
    Args:
        conn (sqlite3.Connection): Active SQLite connection object.
        sql_script_string (str): A string containing one or more SQL statements
                                 separated by semicolons.
    """
    if conn is None:
        print("ERROR: Database connection is not established. Cannot execute SQL script.")
        return
    try:
        cursor = conn.cursor() # A cursor is needed to execute SQL
        cursor.executescript(sql_script_string) # 'executescript' handles multiple statements
        conn.commit() # Save the changes to the database
        print("SQL script executed successfully.")
    except sqlite3.Error as e:
        print(f"Error executing SQL script: {e}\n--- SQL Script Start ---\n{sql_script_string}\n--- SQL Script End ---")

def load_dataframe_to_table(df: pd.DataFrame, 
                             table_name: str, 
                             conn: sqlite3.Connection, 
                             if_exists_action: str = 'replace'):
    """
    Load a Pandas DataFrame into a specified SQL table.

    Args:
        df (pd.DataFrame): The DataFrame to load.
        table_name (str): The name of the target SQL table.
        conn (sqlite3.Connection): Active SQLite connection object.
        if_exists_action (str): What to do if the table already exists.
                                'replace': Drop the table before inserting new values.
                                'append': Insert new values. Old values remain.
                                'fail': Raise a ValueError if table exists.
    """
    if conn is None:
        print(f"ERROR: Database connection not established. Cannot load DataFrame to table '{table_name}'.")
        return
    
    if df.empty:
        # Pandas to_sql with if_exists='replace' on an empty DF will create an empty table
        # For 'append', it does nothing if DF is empty.
        # For 'fail', it also does nothing if DF is empty and table exists.
        # So, it's generally safe, but good to note.
        print(f"INFO: DataFrame for table '{table_name}' is empty. Action: '{if_exists_action}'.")
        if if_exists_action == 'replace':
            try: # Try to drop if exists, then create from empty df's schema
                 df.to_sql(table_name, conn, if_exists=if_exists_action, index=False)
                 print(f"Table '{table_name}' created (or replaced) as empty from DataFrame schema.")
            except Exception as e:
                 print(f"Error creating/replacing empty table '{table_name}': {e}")
        return # Do not proceed if DataFrame is empty and not replacing

    try:
        print(f"Loading data into table '{table_name}' (mode: {if_exists_action})...")
        # index=False means don't write the Pandas DataFrame index as a column in the SQL table
        df.to_sql(table_name, conn, if_exists=if_exists_action, index=False)
        print(f"Successfully loaded {len(df)} rows into '{table_name}'.")
    except Exception as e: # Pandas to_sql can raise various errors
        print(f"ERROR: Could not load DataFrame into table '{table_name}'. Exception: {e}")
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame dtypes:\n", df.dtypes)
        print("First 5 rows of DataFrame to load:\n", df.head().to_string())