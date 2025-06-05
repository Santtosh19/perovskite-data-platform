# src/parsers/elastic_parser.py
import re
import pandas as pd # Still useful to return dict for consistency, could return DataFrame of scalars

def parse_elastic_info_simple(file_path: str) -> dict:
    """
    Parses the Elastic_Info file line-by-line with a simpler, more direct approach.
    Extracts key mechanical properties from the "Average mechanical properties" table
    and specific scalar values using keyword/regex matching.

    Args:
        file_path (str): The path to the Elastic_Info text file.

    Returns:
        dict: A dictionary of extracted properties with standardized keys.
    """

    # 1. Initialize with all expected keys set to None
    extracted_properties = {
        "bulk_modulus_k_hill_gpa": None, "shear_modulus_g_hill_gpa": None, 
        "youngs_modulus_e_hill_gpa": None, "p_wave_modulus_hill_gpa": None,
        "poissons_ratio_v_hill": None, "bulk_shear_ratio_hill": None,
        "isotropic_poissons_ratio": None, "debye_temperature_k": None, 
        "is_mechanically_stable": None, "pugh_ratio": None, 
        "cauchy_pressure_gpa": None, "universal_elastic_anisotropy": None,
        "chung_buessem_anisotropy": None, "longitudinal_wave_velocity_ms": None,
        "transverse_wave_velocity_ms": None, "average_wave_velocity_ms": None,
    }
    
    # For mapping nicer labels, optional
    # label_map = {"GAMMA": "Γ"} # Not really needed for this file, but for concept

    try:
        with open(file_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]

        # State flag to know if we are parsing the main "Average mechanical properties" table
        parsing_average_table = False

        for line_number, current_line in enumerate(lines, 1):
            stripped_line = current_line.strip()

            # Skip fully blank lines
            if not stripped_line:
                continue

            # --- Detect Start of "Average mechanical properties" table ---
            if "Average mechanical properties for bulk polycrystals" in stripped_line:
                parsing_average_table = True
                print(f"DEBUG L{line_number}: Entered Average Properties Table")
                continue # This line is a header, skip to the next line for table data

            # --- Detect End of "Average mechanical properties" table ---
            # The table block effectively ends when we encounter lines for other properties
            # or specific delimiters that are not data rows.
            # A simple way: if we were in the table, but the current line does not
            # look like a table data row OR its known structural elements, assume we've exited.
            if parsing_average_table:
                # Table structure lines (headers, dividers)
                if stripped_line.startswith("+-") or "Scheme" in stripped_line:
                    print(f"DEBUG L{line_number}: Table structure line: '{stripped_line}'")
                    continue # Skip these table structure lines from data row parsing

                # Actual data rows start with "|"
                if stripped_line.startswith("|"):
                    raw_cols = current_line.split('|') # Use original line for split to handle spacing around '|'
                    cleaned_cols = [col.strip() for col in raw_cols]
                    
                    print(f"DEBUG L{line_number}: Table Row Candidate Parts: {cleaned_cols}")

                    if len(cleaned_cols) >= 4: # Need at least ' ', Label, V, R, Hill
                        label_part = cleaned_cols[1].lower() # Make label lowercase for easier matching
                        hill_value_str = cleaned_cols[4]

                        try:
                            hill_value_float = float(hill_value_str)
                            if "bulk modulus k (gpa)" in label_part:
                                extracted_properties["bulk_modulus_k_hill_gpa"] = hill_value_float
                            elif "shear modulus g (gpa)" in label_part:
                                extracted_properties["shear_modulus_g_hill_gpa"] = hill_value_float
                            elif "young's modulus e (gpa)" in label_part: # Careful with apostrophe
                                extracted_properties["youngs_modulus_e_hill_gpa"] = hill_value_float
                            elif "p-wave modulus (gpa)" in label_part:
                                extracted_properties["p_wave_modulus_hill_gpa"] = hill_value_float
                            elif "poisson's ratio v" in label_part and "isotropic" not in label_part:
                                extracted_properties["poissons_ratio_v_hill"] = hill_value_float
                            elif "bulk/shear ratio" in label_part:
                                extracted_properties["bulk_shear_ratio_hill"] = hill_value_float
                        except ValueError:
                            print(f"WARNING (Elastic Table L{line_number}): Could not parse float from Hill value '{hill_value_str}' for label '{cleaned_cols[1]}'")
                    # After processing a table data row, continue to the next line from the file
                    continue 
                else:
                    # If we were in the table, but this line doesn't start with "|" and isn't a structure line,
                    # we must have exited the table's data rows.
                    print(f"DEBUG L{line_number}: Exited Average Properties Table (found non-data line: '{stripped_line}')")
                    parsing_average_table = False 
                    # Now this current line will be processed by the scalar checks below
            
            # --- If NOT in the "Average mechanical properties" table's data rows, check for other scalars ---
            # This block will execute for lines BEFORE the table, AFTER the table,
            # or if a line within the table section wasn't a data row (already handled by `continue`).

            # Stability
            if "This Structure is Mechanically Stable" in stripped_line:
                extracted_properties["is_mechanically_stable"] = True
                continue # Found it, move to next line
            elif "This Structure is Mechanically Unstable" in stripped_line:
                extracted_properties["is_mechanically_stable"] = False
                continue # Found it, move to next line

            # Key-value pairs using regex - only try if property not already found
            # Define simple regex for key: value pattern (value is float)
            # Using current_line to preserve spacing for regex if important
            key_value_float_pattern = r"^(.*?):\s*([+-]?\d+\.\d+)" # Group 1: Key, Group 2: Value

            match = re.search(key_value_float_pattern, current_line.strip()) # Strip for this general pattern
            if match:
                key_candidate = match.group(1).strip().lower() # Get key, strip, lowercase
                value_candidate_str = match.group(2)
                
                try:
                    value_float = float(value_candidate_str)
                    if "isotropic poisson's ratio" == key_candidate and extracted_properties["isotropic_poissons_ratio"] is None:
                        extracted_properties["isotropic_poissons_ratio"] = value_float
                    elif "debye temperature (in k)" == key_candidate and extracted_properties["debye_temperature_k"] is None:
                        extracted_properties["debye_temperature_k"] = value_float
                    elif "pugh ratio" == key_candidate and extracted_properties["pugh_ratio"] is None:
                        extracted_properties["pugh_ratio"] = value_float
                    elif "cauchy pressure (gpa)" == key_candidate and extracted_properties["cauchy_pressure_gpa"] is None:
                        extracted_properties["cauchy_pressure_gpa"] = value_float
                    elif "universal elastic anisotropy" == key_candidate and extracted_properties["universal_elastic_anisotropy"] is None:
                        extracted_properties["universal_elastic_anisotropy"] = value_float
                    elif "chung-buessem anisotropy" == key_candidate and extracted_properties["chung_buessem_anisotropy"] is None: # check spelling
                        extracted_properties["chung_buessem_anisotropy"] = value_float
                    elif "longitudinal wave velocity (in m/s)" == key_candidate and extracted_properties["longitudinal_wave_velocity_ms"] is None:
                        extracted_properties["longitudinal_wave_velocity_ms"] = value_float
                    elif "transverse wave velocity (in m/s)" == key_candidate and extracted_properties["transverse_wave_velocity_ms"] is None:
                        extracted_properties["transverse_wave_velocity_ms"] = value_float
                    elif "average wave velocity (in m/s)" == key_candidate and extracted_properties["average_wave_velocity_ms"] is None:
                        extracted_properties["average_wave_velocity_ms"] = value_float
                except ValueError:
                    print(f"WARNING (Elastic Scalar L{line_number}): Could not convert value '{value_candidate_str}' for key '{key_candidate}'")
                continue # Processed this line as a key-value pair

        # --- Final Post-processing ---
        if extracted_properties.get("poissons_ratio_v_hill") is None and \
           extracted_properties.get("isotropic_poissons_ratio") is not None:
            extracted_properties["poissons_ratio_v_hill"] = extracted_properties["isotropic_poissons_ratio"]
        
        return extracted_properties

    except FileNotFoundError:
        print(f"ERROR (Elastic Parser): File not found at {file_path}")
        return {key: None for key in extracted_properties}
    except Exception as e:
        print(f"ERROR (Elastic Parser): Unexpected error parsing {file_path}: {e}")
        return {key: None for key in extracted_properties}