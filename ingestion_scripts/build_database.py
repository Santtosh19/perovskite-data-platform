# ingestion_scripts/03_build_database.py
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

# Assuming db_manager.py is in src/
from db_manager import create_connection, execute_sql_script, load_dataframe_to_table 

# --- Configuration: Define File Paths ---
BASE_DIR = project_root
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PARSED_COLLECTIONS_DIR = INTERIM_DATA_DIR / "parsed_property_collections" # Where your collection CSVs are

DB_FILE_PATH = str(PROCESSED_DATA_DIR / "perovskite_platform.sqlite")
SUMMARY_PROPERTIES_CSV = PROCESSED_DATA_DIR / "material_summary_properties.csv"

# --- 1. DEFINE YOUR SQL SCHEMA (CREATE TABLE statements) ---
# This string contains all your CREATE TABLE statements.
# DROP TABLE IF EXISTS ensures a clean build each time.
SCHEMA_SQL = """
DROP TABLE IF EXISTS Materials;
CREATE TABLE Materials (
    material_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL UNIQUE,
    material_formula TEXT NOT NULL,
    band_character TEXT,
    direct_gap_ev REAL,
    indirect_gap_ev REAL,
    fermi_energy_ev REAL,
    bulk_modulus_k_hill_gpa REAL,
    shear_modulus_g_hill_gpa REAL,
    youngs_modulus_e_hill_gpa REAL,
    p_wave_modulus_hill_gpa REAL,
    poissons_ratio_v_hill REAL,
    bulk_shear_ratio_hill REAL,
    isotropic_poissons_ratio REAL,
    debye_temperature_k REAL,
    is_mechanically_stable INTEGER, 
    pugh_ratio REAL,
    cauchy_pressure_gpa REAL,
    universal_elastic_anisotropy REAL,
    chung_buessen_anisotropy REAL,
    longitudinal_wave_velocity_ms REAL,
    transverse_wave_velocity_ms REAL,
    average_wave_velocity_ms REAL,
);

DROP TABLE IF EXISTS BandStructureData;
CREATE TABLE BandStructureData ( /* USING LONG FORMAT */
    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL,
    material_formula TEXT NOT NULL,                
    k_distance REAL,
    band_index INTEGER NOT NULL, 
    energy_ev REAL               
    /* FOREIGN KEY (material_x_value) REFERENCES Materials(material_x_value) */
);

DROP TABLE IF EXISTS KLabelsData;
CREATE TABLE KLabelsData (
    klabel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL,
    material_formula TEXT,            
    K_Label TEXT,               
    K_Coordinate REAL           
    /* FOREIGN KEY (material_x_value) REFERENCES Materials(material_x_value) */
);

DROP TABLE IF EXISTS HighSymmetryPointsData;
CREATE TABLE HighSymmetryPointsData (
    hs_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL,
    material_formula TEXT, 
    label TEXT,
    frac_x REAL,
    frac_y REAL,
    frac_z REAL
    /* FOREIGN KEY (material_x_value) REFERENCES Materials(material_x_value) */
);

DROP TABLE IF EXISTS OpticalSpectraData;
CREATE TABLE OpticalSpectraData (
    optical_data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL,
    material_formula TEXT,          
    spectrum_type TEXT NOT NULL,    
    photon_energy_ev REAL,
    val_xx REAL,                    
    val_yy REAL,
    val_zz REAL,
    val_xy REAL,
    val_yz REAL,
    val_zx REAL,
    val_avg REAL                    
    /* FOREIGN KEY (material_x_value) REFERENCES Materials(material_x_value) */
);

DROP TABLE IF EXISTS DensityOfStatesData;
CREATE TABLE DensityOfStatesData (
    dos_data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_x_value INTEGER NOT NULL,
    material_formula TEXT,
    Energy_eV REAL,
    tdos REAL,        
    /* FOREIGN KEY (material_x_value) REFERENCES Materials(material_x_value) */
);
"""

# --- 2. HELPER FUNCTION TO LOAD DATA (generic one can be used now) ---
# (No need for specialized load_spectral_or_band_csv_to_db if using direct names and simple loads)

# --- 3. MAIN DATABASE BUILDING LOGIC ---
def main_db_build():
    print("--- Script 03: Building SQLite Database (Simplified with Long Band Format) ---")
    
    # Ensure output directories exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # PARSED_COLLECTIONS_DIR might not be strictly needed if reading directly from source .dat files later
    # But if 02_extract...py saves CSVs, ensure this directory creation logic is there too.
    
    # Remove old DB for a clean build
    if os.path.exists(DB_FILE_PATH):
        print(f"Removing old database: {DB_FILE_PATH}")
        os.remove(DB_FILE_PATH)

    conn = create_connection(DB_FILE_PATH)
    if conn is None:
        print("Failed to create database connection. Exiting.")
        return

    # Create all tables defined in SCHEMA_SQL
    print("Creating database schema...")
    execute_sql_script(conn, SCHEMA_SQL) # from db_manager.py

    # --- A. Load Materials Summary Data (Scalar Properties) ---
    print(f"\nLoading Materials summary data from: {SUMMARY_PROPERTIES_CSV}")
    if SUMMARY_PROPERTIES_CSV.exists():
        try:
            summary_df = pd.read_csv(SUMMARY_PROPERTIES_CSV)
            if not summary_df.empty:
                # Convert Python booleans from CSV (read as objects/strings) to 0/1 for SQLite INTEGER
                bool_cols_to_convert = [
                    'is_mechanically_stable', 'criteria_c11_gt_c12_met', 
                    'criteria_c11_add_2c12_gt_0_met', 'criteria_c44_gt_0_met'
                ]
                for col in bool_cols_to_convert:
                    if col in summary_df.columns:
                        summary_df[col] = summary_df[col].apply(
                            lambda x: 1 if str(x).lower() == 'true' else (0 if str(x).lower() == 'false' else pd.NA)
                        ).astype('Int64') # Int64 handles NA for integers

                load_dataframe_to_table(summary_df, "Materials", conn, if_exists_action='append')
            else:
                print(f"INFO: Summary CSV '{SUMMARY_PROPERTIES_CSV}' is empty.")
        except Exception as e:
            print(f"ERROR loading Materials summary data: {e}")
    else:
        print(f"ERROR: Summary properties CSV not found: {SUMMARY_PROPERTIES_CSV}")


    # --- B. Load Band Structure Data (Transforming Wide to Long) ---
    print(f"\nLoading Band Structure data...")
    band_structure_csv_path = PARSED_COLLECTIONS_DIR / "all_band_structure_data.csv" # Output from Script 02
    if band_structure_csv_path.exists():
        try:
            wide_band_df = pd.read_csv(band_structure_csv_path)
            if not wide_band_df.empty:
                id_vars = [col for col in ['material_x_value', 'calc_type', 'k_distance', 'material_formula'] if col in wide_band_df.columns]
                value_vars = [col for col in wide_band_df.columns if col.startswith('band_') and col.endswith('_ev')]
                
                if not id_vars or not value_vars: # Basic check
                    print(f"ERROR: Could not identify id_vars or value_vars for melting in {band_structure_csv_path}")
                else:
                    long_band_df = pd.melt(
                        wide_band_df,
                        id_vars=id_vars,
                        value_vars=value_vars,
                        var_name='band_label_str',
                        value_name='energy_ev'
                    )
                    long_band_df['band_index'] = long_band_df['band_label_str'].str.extract(r'band_(\d+)_ev').astype(int)
                    
                    # Select columns for the SQL table (material_formula not in BandStructureData long schema)
                    columns_for_db = ['material_x_value', 'calc_type', 'k_distance', 'band_index', 'energy_ev']
                    # Ensure all selected columns actually exist after melt/extract
                    final_long_band_df = long_band_df[[col for col in columns_for_db if col in long_band_df.columns]]
                    
                    load_dataframe_to_table(final_long_band_df, "BandStructureData", conn, if_exists_action='append')
            else:
                print(f"INFO: Band structure CSV '{band_structure_csv_path}' is empty.")
        except Exception as e:
            print(f"ERROR processing or loading BandStructureData from '{band_structure_csv_path}': {e}")
    else:
        print(f"WARNING: Band structure CSV not found: {band_structure_csv_path}")


    # --- C. Load KLabels, HighSymmetryPoints (Direct from their CSVs) ---
    print(f"\nLoading KLabels data...")
    klabels_csv_path = PARSED_COLLECTIONS_DIR / "all_klabels_data.csv"
    if klabels_csv_path.exists():
        try:
            klabels_df = pd.read_csv(klabels_csv_path)
            if not klabels_df.empty: load_dataframe_to_table(klabels_df, "KLabelsData", conn, if_exists_action='append')
            else: print(f"INFO: KLabels CSV '{klabels_csv_path}' is empty.")
        except Exception as e: print(f"ERROR loading KLabels data: {e}")
    else: print(f"WARNING: KLabels CSV not found: {klabels_csv_path}")

    print(f"\nLoading High Symmetry Points data...")
    hs_points_csv_path = PARSED_COLLECTIONS_DIR / "all_high_symmetry_points_data.csv"
    if hs_points_csv_path.exists():
        try:
            hs_df = pd.read_csv(hs_points_csv_path)
            if not hs_df.empty: load_dataframe_to_table(hs_df, "HighSymmetryPointsData", conn, if_exists_action='append')
            else: print(f"INFO: High Symmetry CSV '{hs_points_csv_path}' is empty.")
        except Exception as e: print(f"ERROR loading High Symmetry Points data: {e}")
    else: print(f"WARNING: High Symmetry Points CSV not found: {hs_points_csv_path}")


    # --- D. Load Optical Spectra Data ---
    print(f"\nLoading Optical Spectra data...")
    optical_files_to_load_map = { # CSV_filename : spectrum_type_label_for_db
        "all_absorption_data.csv": "ABSORPTION",
        "all_reflectivity_data.csv": "REFLECTIVITY",
        "all_refractive_data.csv": "REFRACTIVE_N", # Assuming refractive.dat is 'n'
        # Add: "all_real_epsilon_data.csv": "EPSILON1_REAL", if you parsed real.in
        # Add: "all_imag_epsilon_data.csv": "EPSILON2_IMAG", if you parsed imag.in
    }
    for csv_file, spectrum_label in optical_files_to_load_map.items():
        optical_csv_path = PARSED_COLLECTIONS_DIR / csv_file
        if optical_csv_path.exists():
            try:
                optical_df = pd.read_csv(optical_csv_path)
                if not optical_df.empty:
                    if 'spectrum_type' not in optical_df.columns: # Add if not already there
                        optical_df['spectrum_type'] = spectrum_label
                    # Ensure all columns match the OpticalSpectraData table schema
                    # For example, parse_vaspkit_optic_file outputs val_xx, val_yy etc. which are in the schema
                    # If you added val_avg, ensure it's passed.
                    # If the CSV has more columns than schema, to_sql might fail or create them if no schema used.
                    # Better to select only schema-defined columns from optical_df before loading.
                    # Example, assuming your schema defined 'photon_energy_ev', 'val_xx', ... 'val_avg'
                    cols_for_optical_db = ['material_x_value', 'material_formula', 'spectrum_type', 
                                           'photon_energy_ev', 'val_xx', 'val_yy', 'val_zz',
                                           'val_xy', 'val_yz', 'val_zx', 'val_avg']
                    # Select only columns that exist in BOTH the DataFrame AND our target list
                    final_optical_df_cols = [col for col in cols_for_optical_db if col in optical_df.columns]
                    final_optical_df = optical_df[final_optical_df_cols]

                    load_dataframe_to_table(final_optical_df, "OpticalSpectraData", conn, if_exists_action='append')
                else: print(f"INFO: Optical CSV '{optical_csv_path}' is empty.")
            except Exception as e: print(f"ERROR loading Optical data from '{optical_csv_path}' for type '{spectrum_label}': {e}")
        else: print(f"WARNING: Optical CSV not found: {optical_csv_path}")


    # --- E. Load Density of States Data ---
    print(f"\nLoading Density of States data...")
    tdos_csv_path = PARSED_COLLECTIONS_DIR / "all_tdos_data.csv"
    if tdos_csv_path.exists():
        try:
            tdos_df = pd.read_csv(tdos_csv_path)
            if not tdos_df.empty:
                # Ensure columns match schema: material_x_value, material_formula, energy_ev, total_dos, dos_spin_up, dos_spin_down
                # Your tdos_parser should output these names or you map them here.
                # If dos_spin_up/down are not always present, make sure they are pd.NA or None for those rows.
                cols_for_dos_db = ['material_x_value', 'material_formula', 'Energy_eV', 'tdos']
                final_dos_df_cols = [col for col in cols_for_dos_db if col in tdos_df.columns]
                final_dos_df = tdos_df[final_dos_df_cols]

                load_dataframe_to_table(final_dos_df, "DensityOfStatesData", conn, if_exists_action='append')
            else: print(f"INFO: TDOS CSV '{tdos_csv_path}' is empty.")
        except Exception as e: print(f"ERROR loading TDOS data: {e}")
    else: print(f"WARNING: TDOS CSV not found: {tdos_csv_path}")


    if conn:
        conn.close()
        print(f"\nDatabase connection closed. SQLite DB build at: {DB_FILE_PATH}")
    print("--- Script 03: Database Build Finished ---")

if __name__ == "__main__":
    main_db_build()



'''


'''