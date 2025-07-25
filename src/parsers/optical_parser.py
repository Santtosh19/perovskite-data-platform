# src/parsers/optical_parser.py
import pandas as pd

def parse_vaspkit_optic_file(file_path: str) -> pd.DataFrame:
    """
    Parses multi-column optical data files typically generated by VASP/vaspkit.
    These files usually have one or more header/comment lines (often starting with '#'),
    followed by columns of numerical data (Energy, xx, yy, zz, xy, yz, zx).

    This version uses a simpler approach to skip header lines.

    Args:
        file_path (str): Path to the optical data file (e.g., ABSORPTION.dat).

    Returns:
        pd.DataFrame: Parsed data with standardized column names or an empty DataFrame on error.
    """
    
    # Define expected column names if the full 7-column structure is present
    # (1 energy column + 6 tensor component columns)
    default_column_names = [
        'photon_energy_ev', 
        'val_xx', 'val_yy', 'val_zz', 
        'val_xy', 'val_yz', 'val_zx'
    ]

    try:
        # --- Step 1: Read all lines from the file ---
        with open(file_path, 'r') as f:
            all_lines = f.readlines()

        # --- Step 2: Find the first line that looks like data ---
        # A data line typically starts with a number (the energy value).
        # Header/comment lines usually start with '#' or might be descriptive text.
        data_start_line_index = 0
        for i, line in enumerate(all_lines):#i is the index of the line while line is the content (item) of the line
            stripped_line = line.strip()
            if not stripped_line: # Skip blank lines
                data_start_line_index = i + 1
                continue
            if stripped_line.startswith("#"): # Skip VASP comment lines
                data_start_line_index = i + 1
                continue
            
            # Try to see if the first "word" on the line can be a float
            first_token = stripped_line.split(maxsplit=1)[0] if stripped_line else ""
            try:
                float(first_token)
                # If successful, this is likely the first data line
                break # Exit loop, data_start_line_index is set correctly
            except ValueError:
                # If not a float, it's probably still part of a header or an unexpected line
                data_start_line_index = i + 1
            
            if i > 10: # Safety: if we scan too many lines without finding data, assume something is wrong or format is very unusual
                print(f"Warning: Scanned more than 10 lines in {file_path} without finding a clear data start. Will attempt to read from line {data_start_line_index}.")
                break
        
        # --- Step 3: Create a list of only the data lines ---
        data_lines_to_parse = []
        if data_start_line_index < len(all_lines):
            for line in all_lines[data_start_line_index:]:
                stripped_line = line.strip()
                # Further filter: ensure data lines are not just leftover comments
                if stripped_line and not stripped_line.startswith("#"):
                    data_lines_to_parse.append(stripped_line)
        
        if not data_lines_to_parse:
            print(f"Warning: No data lines found in {file_path} after attempting to skip headers.")
            return pd.DataFrame() # Return empty DataFrame

        # --- Step 4: Use pandas to parse the identified data lines ---
        # We create a "virtual file" from our list of data strings using io.StringIO
        from io import StringIO
        data_as_string_buffer = StringIO("\n".join(data_lines_to_parse))

        df = pd.read_csv(
            data_as_string_buffer, # Read from our string buffer
            sep=r'\s+',            # One or more whitespace characters
            header=None           # No header in the data_lines_to_parse
        )

        # --- Step 5: Assign column names ---
        num_cols_read = df.shape[1]
        actual_column_names = []

        if num_cols_read > 0:
            # Take as many names from default_column_names as there are columns
            actual_column_names = default_column_names[:num_cols_read]
            # If more columns were read than we have default names, add generic ones
            if num_cols_read > len(default_column_names):
                for i in range(len(default_column_names), num_cols_read):
                    actual_column_names.append(f'extra_col_{i - len(default_column_names) + 1}')
        
        if actual_column_names:
            df.columns = actual_column_names
        else: # Should only happen if df was empty initially
            print(f"Warning: DataFrame for {file_path} was empty after read_csv, cannot assign column names.")
            return pd.DataFrame()


        return df

    except pd.errors.EmptyDataError: # This might still be triggered by StringIO if data_lines_to_parse is empty
        print(f"Warning: Optical data file {file_path} resulted in no data for pandas after header skipping.")
        return pd.DataFrame() 
    except FileNotFoundError:
        print(f"ERROR: Optical data file {file_path} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing VASPkit-style optical data file {file_path}: {e}")
        return pd.DataFrame()