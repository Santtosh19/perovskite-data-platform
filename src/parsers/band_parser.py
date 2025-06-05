# src/parsers/band_parser.py
import pandas as pd # Import the pandas library, which is great for working with table-like data
import re # For regular expressions

def parse_reformatted_band_dat(file_path):
    """
    Parses REFORMATTED_BAND.dat.
    This file usually contains:
    - A header line (often starting with '#') that we need to ignore.
    - Columns of numbers separated by spaces.
    - The first column is the 'k-path distance' (like an x-axis for the plot).
    - The following columns are the energy values for different 'bands' (like multiple y-values for each x).
    """
    try: # This 'try-except' block is for error handling. If something goes wrong during parsing, 
         # the program won't just crash; it will print an error and return an empty table.

        # Use pandas to read the CSV-like data from the file.
        # Even though it's a .dat file, if it's text with columns separated by spaces,
        # pandas can treat it like a CSV (Comma Separated Values) or TSV (Tab Separated Values) file.
        df = pd.read_csv( 
            file_path,          # This is the path to the file you want to read (e.g., "data/raw/.../REFORMATTED_BAND.dat")
            
            sep=r'\s+',         # 'sep' tells pandas what character separates the values in each row.
                                # r'\s+' is a "regular expression" that means "one or more whitespace characters".
                                # This is good because sometimes there might be one space, sometimes multiple.
                                # If you KNEW it was always a single space, you could use sep=' '. If tabs, sep='\t'.
            
            comment='#',        # This tells pandas to ignore any line that starts with a '#' symbol.
                                # Your REFORMATTED_BAND.dat example had "#K-Path Band-1..." as the first line.
                                # So, pandas will automatically skip this header line for us.
            
            header=None         # 'header=None' tells pandas that the file DOES NOT have a header row 
                                # that it should use to automatically name the columns of our DataFrame.
                                # We are skipping the actual header with 'comment=#' and will name columns ourselves.
        )

        # After reading, 'df' is now a Pandas DataFrame (like a table or spreadsheet in Python).
        # Now we need to give meaningful names to the columns.

        # How many bands are there?
        # df.shape gives (number_of_rows, number_of_columns).
        # df.shape[1] is the total number of columns pandas read.
        # The first column is 'k_distance', so the rest are bands.
        num_bands = df.shape[1] - 1 

        # Create a list of column names
        # Start with 'k_distance' for the first column.
        # Then, for each band, create a name like 'band_1_ev', 'band_2_ev', etc.
        # The '_ev' is added to remind us the energy is in electron-Volts (a common unit).
        column_names = ['k_distance'] + [f'band_{i+1}_ev' for i in range(num_bands)]
        #   - f'band_{i+1}_ev' is an "f-string" for formatted string literals.
        #   - range(num_bands) goes from 0 up to (num_bands - 1).
        #   - So i+1 makes the band numbers start from 1 (band_1, band_2...).

        # Check if the number of names we created matches the number of columns in our DataFrame.
        # This is a safety check.
        if len(column_names) == len(df.columns):
            df.columns = column_names # Assign our list of names to be the column headers of the DataFrame.
        else:
            # If the numbers don't match, something went wrong (e.g., pandas read an extra empty column).
            print(f"Warning: Column name mismatch for {file_path}. Expected {len(column_names)} cols, got {len(df.columns)} in data.")
            print("This might be due to trailing spaces or inconsistent column numbers in the file.")
            # If the DataFrame is empty or has too few columns to be useful, return an empty one.
            if df.empty or df.shape[1] <= 1: # df.shape[1] <= 1 means only k_distance or less, no band data.
                 print(f"Error: Not enough columns read to assign names correctly in {file_path}.")
                 return pd.DataFrame() # Return an empty DataFrame

        return df # Return the nice, structured DataFrame

    except pd.errors.EmptyDataError: # Specific error if pandas finds the file empty (or only comments)
        print(f"Warning: File {file_path} is empty or contains only comments after skipping.")
        return pd.DataFrame() # Return an empty DataFrame

    except Exception as e: # Catch any other unexpected errors during parsing.
        print(f"Error parsing REFORMATTED_BAND.dat file {file_path}: {e}")
        return pd.DataFrame() # Return an empty DataFrame
    
def parse_high_symmetry_points_fractional_manual(file_path): # Renamed for clarity
    """
    Parses HIGH_SYMMETRY_POINTS file line-by-line.
    This file contains fractional coordinates and labels for high-symmetry points.
    Returns a Pandas DataFrame with columns: ['label', 'frac_x', 'frac_y', 'frac_z'].
    """
    points_data_rows = []
    label_map = {"GAMMA": "Γ"} # Extend as needed based on labels in your files

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Identify where actual data starts by skipping known header patterns
        # and ignoring known footer patterns.
        data_lines_started = False
        for single_line_text in lines:
            cleaned_line = single_line_text.strip()

            # Skip empty lines
            if not cleaned_line:
                continue

            # Footer detection (specific to this file format)
            if "If you use this module, please cite" in cleaned_line:
                break # Stop processing if we hit the footer

            # Header detection (more flexible than fixed line count)
            if not data_lines_started:
                if "High-symmetry points" in cleaned_line or \
                   "You can check them" in cleaned_line or \
                   cleaned_line.startswith("#"): # General comment skip
                    continue # Skip this header/comment line
                else:
                    # If it's not a known header/comment and not empty, assume it's a data line
                    data_lines_started = True 
            
            if not data_lines_started: # Still in header/comment phase
                continue

            # Now process the presumed data line
            parts = cleaned_line.split()
            if len(parts) == 4: # Expecting: frac_x frac_y frac_z LABEL
                try:
                    frac_x = float(parts[0])
                    frac_y = float(parts[1])
                    frac_z = float(parts[2])
                    label_str = parts[3]
                    
                    #mapped_label = label_map.get(label_str.upper(), label_str)
                    
                    current_row_data = {
                        'label': label_str,
                        'frac_x': frac_x,
                        'frac_y': frac_y,
                        'frac_z': frac_z
                    }
                    points_data_rows.append(current_row_data)
                except ValueError:
                    print(f"WARNING: Could not convert coordinates to float in HIGH_SYMMETRY_POINTS file '{file_path}' on line: '{cleaned_line}'")
            elif cleaned_line: # If line is not empty but doesn't have 4 parts, it's unexpected
                print(f"WARNING: Line in HIGH_SYMMETRY_POINTS file '{file_path}' has unexpected format: '{cleaned_line}'")
        
        if points_data_rows:
            return pd.DataFrame(points_data_rows)
        else:
            return pd.DataFrame(columns=['label', 'frac_x', 'frac_y', 'frac_z'])

    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame(columns=['label', 'frac_x', 'frac_y', 'frac_z'])
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while parsing HIGH_SYMMETRY_POINTS file {file_path}: {e}")
        return pd.DataFrame(columns=['label', 'frac_x', 'frac_y', 'frac_z'])
    
def parse_klabels(file_path): # Renamed for clarity
    points_data_rows = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Identify where actual data starts by skipping known header patterns
        # and ignoring known footer patterns.
        data_lines_started = False
        for single_line_text in lines:
            cleaned_line = single_line_text.strip()

            # Skip empty lines
            if not cleaned_line:
                continue

            # Footer detection (specific to this file format)
            if "* Give the label" in cleaned_line:
                break # Stop processing if we hit the footer

            # Header detection (more flexible than fixed line count)
            if not data_lines_started:
                if "K-Label" in cleaned_line or \
                   "K-Coordinate in band-structure plots " in cleaned_line or \
                   cleaned_line.startswith("#"): # General comment skip
                    continue # Skip this header/comment line
                else:
                    # If it's not a known header/comment and not empty, assume it's a data line
                    data_lines_started = True 
            
            if not data_lines_started: # Still in header/comment phase
                continue

            # Now process the presumed data line
            parts = cleaned_line.split()
            if len(parts) == 2: 
                try:
                    #K_Label = parts[0]/ wont work if label has spaces
                    #K_coord = float(parts[1])/wont work if coord has spaces
                    K_coord = parts[-1]#always grabs the last item (which we assume is the number).
                    K_Label = " ".join(parts[:-1])#always grabs everything except the last item.
                
                    
                    #mapped_label = label_map.get(label_str.upper(), label_str)
                    
                    current_row_data = {
                        'K-Label': K_Label,
                        'K-Coordinate in band-structure plots': K_coord
                    }
                    points_data_rows.append(current_row_data)
                except ValueError:
                    print(f"WARNING: Could not convert coordinates to float in HIGH_SYMMETRY_POINTS file '{file_path}' on line: '{cleaned_line}'")
            elif cleaned_line: # If line is not empty but doesn't have 2 parts, it's unexpected
                print(f"WARNING: Line in HIGH_SYMMETRY_POINTS file '{file_path}' has unexpected format: '{cleaned_line}'")
        
        if points_data_rows:
            return pd.DataFrame(points_data_rows)
        else:
            return pd.DataFrame(columns=['K-Label', 'K-Coordinate in band-structure plots'])

    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame(columns=['K-Label', 'K-Coordinate in band-structure plots'])
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while parsing K-Label file {file_path}: {e}")
        return pd.DataFrame(columns=['K-Label', 'K-Coordinate in band-structure plots'])

def parse_band_gap_summary_vaspkit(file_path: str) -> dict: # Type hint indicates it SHOULD return a dict
    """
    Parses a BAND_GAP summary file, likely generated by VASPkit or similar,
    to extract detailed band gap information including character, value,
    VBM, CBM, Fermi energy, and HOMO/LUMO band indices.

    Args:
        file_path (str): Path to the BAND_GAP summary file.

    Returns:
        dict: A dictionary with extracted properties.
              Returns keys with None values if properties are not found or parsing fails.
    """
    # Initialize with all expected keys for this specific file format
    # Naming keys to reflect they are from this "gap summary" file initially
    gap_summary = {
        "gapfile_band_character": None, 
        "gapfile_band_gap_ev": None,    
        "gapfile_vbm_ev": None,
        "gapfile_cbm_ev": None, # <<< TYPO: Was "gap_file_cbm_ev" in one of my previous versions, make sure it matches your patterns_to_search key. Corrected here.
        "gapfile_fermi_energy_ev": None,
        "gapfile_homo_band_idx": None,
        "gapfile_lumo_band_idx": None,
    }

    patterns_to_search = [
        ("gapfile_band_character", r"Band Character:\s*(\w+)", str, None),
        ("gapfile_band_gap_ev", r"Band Gap \(eV\):\s*([+-]?\d+\.\d+)", float, None),
        ("gapfile_vbm_ev", r"Eigenvalue of VBM \(eV\):\s*([+-]?\d+\.\d+)", float, None),
        ("gapfile_cbm_ev", r"Eigenvalue of CBM \(eV\):\s*([+-]?\d+\.\d+)", float, None), # <<< Key used here
        ("gapfile_fermi_energy_ev", r"Fermi Energy \(eV\):\s*([+-]?\d+\.\d+)", float, None),
        ("gapfile_homo_lumo_bands", r"HOMO & LUMO Bands:\s*(\d+)\s+(\d+)", (int, int), None),
    ]

    try:
        with open(file_path, 'r') as f:
            for line in f: # Iterating through each line of the file
                line_for_processing = line.strip() # Clean the line once for checks
                if not line_for_processing or line_for_processing.startswith("+---") or "Summary" in line_for_processing : 
                    # Skip delimiter lines like "+----- Summary ------+" or "+-----------------+"
                    # and empty lines or lines only containing "Summary" as part of delimiter
                    continue

                # Try to match each defined pattern against the current line
                for dict_key_config, pattern, value_type_config, flags_config in patterns_to_search:
                    
                    # Optimization: If this property already found, don't try to parse it again
                    already_found = False
                    if dict_key_config == "gapfile_homo_lumo_bands":
                        if gap_summary.get("gapfile_homo_band_idx") is not None and \
                           gap_summary.get("gapfile_lumo_band_idx") is not None:
                            already_found = True
                    elif gap_summary.get(dict_key_config) is not None: # Use .get() for safety
                        already_found = True
                    
                    if already_found:
                        continue # Skip to the next pattern for this line

                    search_flags = flags_config if flags_config is not None else 0
                    # Use original line (line) for re.search if pattern relies on precise internal spacing.
                    # If patterns are robust to internal spacing (e.g. use \s*), line_for_processing is fine.
                    # Let's assume patterns are robust enough for line_for_processing here.
                    match = re.search(pattern, line_for_processing, search_flags)

                    if match:
                        try:
                            if dict_key_config == "gapfile_homo_lumo_bands":
                                gap_summary["gapfile_homo_band_idx"] = int(match.group(1))
                                gap_summary["gapfile_lumo_band_idx"] = int(match.group(2))
                            elif isinstance(value_type_config, tuple): # e.g., for (float, float, float)
                                values = [value_type_config[i](match.group(i+1)) for i in range(len(value_type_config))]
                                gap_summary[dict_key_config] = values
                            else: # Single value expected (str, float, int)
                                value_str = match.group(1) # Value from the first capturing group
                                gap_summary[dict_key_config] = value_type_config(value_str)
                            
                            print(f"  DEBUG -> Matched '{dict_key_config}' with value: {gap_summary.get(dict_key_config)}")
                            break # Found a match for THIS LINE, process next line in file
                        except (ValueError, TypeError, IndexError) as e:
                            print(f"WARNING (BAND_GAP File): Error converting/assigning for '{dict_key_config}' from line '{line_for_processing}' in '{file_path}'. Regex matched: '{match.groups() if match else 'No Match'}'. Error: {e}")
                            # This break is important: if a pattern matches but value conversion fails,
                            # we should probably not try other patterns on this same bad line segment.
                            break 
        
        print(f"DEBUG: Final parsed dictionary for {file_path}: {gap_summary}") # Good for final check
        return gap_summary # Success, return the populated (or partially populated) dictionary

    except FileNotFoundError:
        print(f"ERROR (BAND_GAP File): File '{file_path}' not found.")
        # On critical error like file not found, return the initialized dictionary (all Nones)
        # to maintain a consistent return type for the calling function.
        return gap_summary 
    except Exception as e:
        print(f"ERROR (BAND_GAP File): An unexpected error occurred while parsing '{file_path}': {e}")
        return gap_summary # Consistent return type