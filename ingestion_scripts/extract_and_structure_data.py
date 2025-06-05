# ingestion_scripts/02_extract_and_structure_data.py
import pandas as pd
import pathlib
import sys
import os

# --- Add project root's 'src' directory to Python's path ---
current_script_dir = pathlib.Path(__file__).resolve().parent
project_root = current_script_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Import your working parser modules ---
from parsers import band_parser        
from parsers import optical_parser     
from parsers import elastic_parser     
from parsers import dos_parser         
from parsers import scalar_property_parser

# --- Configuration: Define File Paths ---
BASE_DIR = project_root
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PARSED_COLLECTIONS_DIR = INTERIM_DATA_DIR / "parsed_property_collections"

INVENTORY_FILE = INTERIM_DATA_DIR / "file_inventory.csv"
SUMMARY_PROPERTIES_FILE = PROCESSED_DATA_DIR / "material_summary_properties.csv"

def main():
    print("--- Script 02: Extracting and Structuring Data ---")
    if not INVENTORY_FILE.exists():
        print(f"ERROR: Inventory file not found: {INVENTORY_FILE}. Run 01 script first.")
        return
    try:
        inventory_df = pd.read_csv(INVENTORY_FILE)
    except Exception as e:
        print(f"ERROR loading inventory: {e}")
        return
    if inventory_df.empty:
        print("Inventory is empty.")
        return

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_scalar_material_data = [] 

    all_reformatted_band_data = []
    all_klabels_data = []
    all_high_symmetry_data = []
    all_absorption_data = []
    all_reflectivity_data = []
    all_refractive_data = []
    all_tdos_data = []

    def get_path_from_group(material_file_group, prop_type_folder, file_name_target):
        row = material_file_group[
            (material_file_group['potential_property_type'].str.upper() == prop_type_folder.upper()) & # Made prop_type_folder match case-insensitive
            (material_file_group['file_name'] == file_name_target)
        ]
        if not row.empty:
            return row.iloc[0]['file_path_absolute']
        return None

    for (x_val, formula), material_group_df in inventory_df.groupby(['material_x_value', 'material_formula']):
        if pd.isna(x_val) or pd.isna(formula): continue
        try: material_x_value = int(x_val)
        except ValueError: continue
        
        print(f"\nProcessing Material: x={material_x_value}, Formula='{formula}'")
        current_material_scalars = {'material_x_value': material_x_value, 'material_formula': formula}

        # VVVVVV --- THIS IS THE NEW/CORRECTED BLOCK FOR SCALAR BAND_GAP and FERMI_ENERGY --- VVVVVV
        # --- Parse BAND_GAP summary file (from BAND folder) ---
        # This file contains direct_gap, indirect_gap (via character), and often a consistent Fermi level
        band_gap_summary_path = get_path_from_group(material_group_df, 'BAND', 'BAND_GAP')
        if band_gap_summary_path:
            # Use your specific parser for this file, e.g., parse_band_gap_summary_vaspkit
            parsed_gap_summary = band_parser.parse_band_gap_summary_vaspkit(band_gap_summary_path) 
            
            if parsed_gap_summary:
                # Map the parsed values to your main scalar dictionary keys
                char = parsed_gap_summary.get("gapfile_band_character")
                gap_val_from_file = parsed_gap_summary.get("gapfile_band_gap_ev") # This is the main reported gap

                current_material_scalars["band_character"] = char # Store the character itself
                
                if char == "Direct" and gap_val_from_file is not None:
                    current_material_scalars["direct_gap_ev"] = gap_val_from_file
                    current_material_scalars["indirect_gap_ev"] = gap_val_from_file 
                elif char == "Indirect" and gap_val_from_file is not None:
                    current_material_scalars["indirect_gap_ev"] = gap_val_from_file
                    # For direct_gap_ev in an indirect material, the situation is complex.
                    # The 'gapfile_band_gap_ev' IS the indirect gap.
                    # We might try to use VBM and CBM eigenvalues IF they are at the same k-point (e.g., Gamma for many perovskites)
                    # For simplicity now, let's set direct_gap_ev to None or the same as indirect for now
                    # You might have VBM/CBM eigenvalues from this parser too if you added them:
                    vbm_val = parsed_gap_summary.get("gapfile_vbm_ev")
                    cbm_val = parsed_gap_summary.get("gapfile_cbm_ev")
                    if vbm_val is not None and cbm_val is not None:
                        # This isn't necessarily the smallest direct gap, but is a fundamental gap value
                        current_material_scalars["fundamental_gap_vbm_cbm"] = cbm_val - vbm_val
                    current_material_scalars["direct_gap_ev"] = None # Or gap_val_from_file, if you assume a direct transition is always present
                elif gap_val_from_file is not None: # If character is unknown but gap value exists
                    current_material_scalars["direct_gap_ev"] = gap_val_from_file 
                    current_material_scalars["indirect_gap_ev"] = gap_val_from_file # Default assumption
                
                # Prioritize Fermi energy from this BAND_GAP summary file
                fermi_from_gapfile = parsed_gap_summary.get("gapfile_fermi_energy_ev")
                if fermi_from_gapfile is not None:
                    current_material_scalars["fermi_energy_ev"] = fermi_from_gapfile
                    print(f"  INFO (x={material_x_value}): Using Fermi energy from BAND_GAP summary file: {fermi_from_gapfile}")
        
        # Fallback: Parse standalone FERMI_ENERGY file if not found in BAND_GAP summary
        if current_material_scalars.get("fermi_energy_ev") is None:
            # Try 'BAND' folder first, then 'DOS' for FERMI_ENERGY file
            fermi_path_standalone = get_path_from_group(material_group_df, 'BAND', 'FERMI_ENERGY')
            if not fermi_path_standalone: 
                fermi_path_standalone = get_path_from_group(material_group_df, 'DOS', 'FERMI_ENERGY')
            
            if fermi_path_standalone:
                # Ensure you have 'parse_fermi_energy_file' in scalar_property_parser.py
                fermi_dict_standalone = scalar_property_parser.parse_fermi_energy_file(fermi_path_standalone) 
                if fermi_dict_standalone and fermi_dict_standalone.get('fermi_energy_ev') is not None:
                    current_material_scalars['fermi_energy_ev'] = fermi_dict_standalone.get('fermi_energy_ev')
                    print(f"  INFO (x={material_x_value}): Using Fermi energy from standalone FERMI_ENERGY file: {current_material_scalars['fermi_energy_ev']}")
            else:
                print(f"  WARNING (x={material_x_value}): Fermi energy NOT found in BAND_GAP summary OR standalone FERMI_ENERGY file.")
        # ^^^^^^ --- END OF THE NEW/CORRECTED BLOCK --- ^^^^^^

        # Detailed Band Data: REFORMATTED_BAND.dat (from BAND folder)
        rb_path = get_path_from_group(material_group_df, 'BAND', 'REFORMATTED_BAND.dat')
        if rb_path:
            df = band_parser.parse_reformatted_band_dat(rb_path)
            if not df.empty:
                df['material_x_value'] = material_x_value
                df['material_formula'] = formula # Add formula
                df['calc_type'] = 'GGA' # Assuming BAND folder is GGA
                all_reformatted_band_data.append(df)
        
        # Detailed Band Data: KLABELS
        kl_path = get_path_from_group(material_group_df, 'BAND', 'KLABELS')
        if kl_path:
            df = band_parser.parse_klabels(kl_path) 
            if not df.empty:
                df['material_x_value'] = material_x_value
                df['material_formula'] = formula
                df['calc_type'] = 'GGA'
                all_klabels_data.append(df)

        # Detailed Band Data: HIGH_SYMMETRY_POINTS
        hsp_path = get_path_from_group(material_group_df, 'BAND', 'HIGH_SYMMETRY_POINTS')
        if hsp_path:
            df = band_parser.parse_high_symmetry_points_fractional_manual(hsp_path) # or your chosen parser
            if not df.empty:
                df['material_x_value'] = material_x_value
                df['material_formula'] = formula
                df['calc_type'] = 'GGA'
                all_high_symmetry_data.append(df)

        # --- OPTIC Properties ---
        optic_prop_type = 'OPTIC' # Default
        # (Your existing logic to find if folder is 'OPTIC' or 'Optic' or 'OPTICAL')
        if material_group_df[material_group_df['potential_property_type'] == 'OPTIC'].empty:
            if not material_group_df[material_group_df['potential_property_type'] == 'Optic'].empty:
                optic_prop_type = 'Optic'
            elif not material_group_df[material_group_df['potential_property_type'] == 'OPTICAL'].empty: # Check if 'OPTICAL' from inventory is a possible type
                optic_prop_type = 'OPTICAL'

        optical_files_to_parse = { # Filename : target_list
            'ABSORPTION.dat': all_absorption_data,
            'REFLECTIVITY.dat': all_reflectivity_data,
            'REFRACTIVE.dat': all_refractive_data,
        }
        for filename, target_list in optical_files_to_parse.items():
            optic_file_path = get_path_from_group(material_group_df, optic_prop_type, filename)
            if optic_file_path:
                df = optical_parser.parse_vaspkit_optic_file(optic_file_path)
                if not df.empty:
                    df['material_x_value'] = material_x_value
                    df['material_formula'] = formula
                    target_list.append(df)
        
        # --- DOS Properties ---
        tdos_path = get_path_from_group(material_group_df, 'DOS', 'tdos.dat')
        if tdos_path:
            # Assuming dos_parser.py has parse_tdos_dat similar to how optical parser works
            df = dos_parser.parse_dos(tdos_path) 
            if not df.empty:
                df['material_x_value'] = material_x_value
                df['material_formula'] = formula
                all_tdos_data.append(df)

        # --- ELASTIC Properties (Scalar) ---
        elastic_path = None
        # (Your existing robust logic for finding the elastic file...)
        elastic_files_in_group = material_group_df[
            material_group_df['potential_property_type'].str.upper() == 'ELASTIC'
        ]
        if not elastic_files_in_group.empty:
            exact_match_row = elastic_files_in_group[elastic_files_in_group['file_name'] == 'elastic_info'] 
            if not exact_match_row.empty:
                elastic_path = exact_match_row.iloc[0]['file_path_absolute']
            else: 
                pattern_match_row = elastic_files_in_group[
                    elastic_files_in_group['file_name'].str.contains('elastic', case=False, na=False)
                ]
                if not pattern_match_row.empty:
                    elastic_path = pattern_match_row.iloc[0]['file_path_absolute']
        
        if elastic_path:
            elastic_dict = elastic_parser.parse_elastic_info_simple(elastic_path) # Or your latest elastic parser name
            if elastic_dict:
                current_material_scalars.update(elastic_dict)
        # else:
            # print(f"  WARNING: No suitable Elastic data file found for x={material_x_value}.")
        
        # This append happens ONCE per material, after trying to get all its scalar properties
        all_scalar_material_data.append(current_material_scalars)
        print(f"Finished x={material_x_value}. Scalar Keys collected: {list(current_material_scalars.keys())}")

    # --- Consolidate and Save Master CSVs (This part is outside the material loop) ---
    if all_scalar_material_data:
        summary_df = pd.DataFrame(all_scalar_material_data)
        cols_order = ['material_x_value', 'material_formula'] + \
                     [c for c in summary_df.columns if c not in ['material_x_value', 'material_formula']]
        # Ensure only existing columns are used for reordering
        cols_order_existing = [col for col in cols_order if col in summary_df.columns]
        summary_df = summary_df[cols_order_existing]
        summary_df = summary_df.sort_values(by='material_x_value').reset_index(drop=True)
        
        SUMMARY_PROPERTIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(SUMMARY_PROPERTIES_FILE, index=False)
        print(f"\nSaved Material Summary to: {SUMMARY_PROPERTIES_FILE}")
        print("Summary DF Head:\n", summary_df.head().to_string()) # .to_string() for better console output of DF
    else:
        print("No scalar data was parsed to save to summary file.")

    def save_consolidated_df(df_list, filename_base): # Helper function (remains same)
        if df_list: # ... (same saving logic as before)
            master_df = pd.concat(df_list, ignore_index=True)
            output_path = PARSED_COLLECTIONS_DIR / f"{filename_base}.csv"
            PARSED_COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True) 
            master_df.to_csv(output_path, index=False)
            print(f"Saved {filename_base} to: {output_path}")

    # Call save_consolidated_df for all your lists
    save_consolidated_df(all_reformatted_band_data, "all_band_structure_data") # Assuming calc_type 'GGA' is implied now
    save_consolidated_df(all_klabels_data, "all_klabels_data")
    save_consolidated_df(all_high_symmetry_data, "all_high_symmetry_points_data")
    save_consolidated_df(all_absorption_data, "all_absorption_data")
    save_consolidated_df(all_reflectivity_data, "all_reflectivity_data")
    save_consolidated_df(all_refractive_data, "all_refractive_data")
    save_consolidated_df(all_tdos_data, "all_tdos_data")
    
    print("\n--- Script 02: Finished ---")

if __name__ == "__main__":
    main()