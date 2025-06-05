import os
import pathlib
import pandas as pd
import re # For more complex pattern matching if needed

# --- Configuration ---
# RAW_DATA_CONTAINS_UNZIPPED_CONTENT is the folder that *directly contains* Cs2NaTlCl6, Cs2NaTlBr6, etc.
# This should be 'HBD_Mixed_anions' if that's the folder created inside 'raw' after your manual unzip
RAW_CONTENT_ROOT_NAME = "HBD_Mixed_anions" # The name of the main folder containing all material subfolders
BASE_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR_CONTAINING_CONTENT = BASE_DATA_DIR / "raw" / RAW_CONTENT_ROOT_NAME

INTERIM_DATA_DIR = BASE_DATA_DIR / "interim"
INVENTORY_FILE_NAME = "file_inventory.csv"


def identify_material_from_path_parts(folder_path_parts):
    """
    Identifies material composition (x_value, formula) based on folder path parts.
    folder_path_parts: A list of directory names leading to the data file,
                       relative to the RAW_DATA_DIR_CONTAINING_CONTENT.
                       e.g., ['Cs2NaTlCl6', 'OPTIC'] or ['ClxBr6-xData33', 'Cl2Br4', 'BAND']
    """
    x_value = None
    formula = None
    potential_property_type = "Unknown"

    if not folder_path_parts:
        return None, None, "Unknown"

    # Level 1 folder (e.g., 'Cs2NaTlCl6', 'ClxBr6-xData33')
    level1_folder = folder_path_parts[0]

    if level1_folder == "Cs2NaTlCl6":
        x_value = 6
        formula = "Cs2NaTlCl6"
        if len(folder_path_parts) > 1:
            potential_property_name = folder_path_parts[1] # e.g., OPTIC, BAND
            potential_property_type = potential_property_name.upper()
    elif level1_folder == "Cs2NaTlBr6":
        x_value = 0
        formula = "Cs2NaTlBr6"
        if len(folder_path_parts) > 1:
            potential_property_name = folder_path_parts[1]
            potential_property_type = potential_property_name.upper()
    elif level1_folder == "ClxBr6-x_Data33": # Contains x=2, 4
        if len(folder_path_parts) > 1:
            level2_folder = folder_path_parts[1] # e.g., 'Cl2Br4', 'Cl4Br2'
            if level2_folder == "Cl2Br4":
                x_value = 2
                formula = "Cs2NaTlCl2Br4"
            elif level2_folder == "Cl4Br2":
                x_value = 4
                formula = "Cs2NaTlCl4Br2"
            if len(folder_path_parts) > 2:
                potential_property_name = folder_path_parts[2] # e.g., OPTIC, BAND
                potential_property_type = potential_property_name.upper()
    elif level1_folder == "ClxBr1-x_Part4": # Contains x=1, 3, 5 (Typo likely in Dryad README, might be ClxBr6-x)
        if len(folder_path_parts) > 1:
            level2_folder = folder_path_parts[1] # e.g., 'ClBr5', 'Cl3Br3', 'Cl5Br'
            if level2_folder == "ClBr5":
                x_value = 1
                formula = "Cs2NaTlClBr5"
            elif level2_folder == "Cl3Br3":
                x_value = 3
                formula = "Cs2NaTlCl3Br3"
            elif level2_folder == "Cl5Br":
                x_value = 5
                formula = "Cs2NaTlCl5Br" # Or Cs2NaTlCl5Br1
            if len(folder_path_parts) > 2:
                potential_property_name = folder_path_parts[2]
                potential_property_type = potential_property_name.upper()
    else:
        # This case might occur if files are directly in the root of RAW_DATA_DIR_CONTAINING_CONTENT
        # or if the structure is unexpected. We can try to guess property type from parent.
        if len(folder_path_parts) > 0: # i.e. not in the root RAW_DATA_DIR_CONTAINING_CONTENT
            potential_property_name = folder_path_parts[-1]
            potential_property_type = potential_property_name.upper() # last folder before file


    return x_value, formula, potential_property_type


def create_file_inventory(content_root_path, inventory_output_path):
    """Scans the content directory and creates an inventory of .dat and OUTCAR files."""
    print(f"Creating file inventory for {content_root_path}...")
    file_list = []

    if not content_root_path.exists():
        print(f"ERROR: Content root path {content_root_path} does not exist. Check RAW_DATA_DIR_CONTAINING_CONTENT setting.")
        return

    for file_path_obj in content_root_path.rglob('*'): # rglob finds files in subdirectories
        if file_path_obj.is_file():
            file_name = file_path_obj.name
            # Target files based on Dryad README descriptions
            if file_name.endswith(".dat") or \
               file_name == "OUTCAR" or \
               file_name == "BAND_GAP" or \
               file_name.startswith("elastic") or \
               file_name.endswith(".in") or \
               file_name.startswith("Elastic") or \
               file_name in ["KLABELS", "HIGH_SYMMETRY_POINTS"]:

                # Get relative path parts to identify material and property
                # Relative to the `content_root_path`
                relative_path_to_file = file_path_obj.relative_to(content_root_path)
                # folder_path_parts are the directory components leading to the file
                folder_path_parts = list(relative_path_to_file.parent.parts)

                x_value, formula, potential_property_type = identify_material_from_path_parts(folder_path_parts)

                file_list.append({
                    "file_path_absolute": str(file_path_obj.resolve()), # Store absolute for direct access
                    "file_path_relative_to_content_root": str(relative_path_to_file),
                    "material_x_value": x_value,
                    "material_formula": formula,
                    "potential_property_type": potential_property_type,
                    "file_name": file_name
                })

    if not file_list:
        print("No relevant files found. Check rglob pattern or directory structure at {content_root_path}.")
        return

    # Create interim directory if it doesn't exist
    inventory_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    inventory_df = pd.DataFrame(file_list)
    inventory_df.to_csv(inventory_output_path, index=False)
    print(f"File inventory saved to {inventory_output_path}")
    print(f"Found {len(inventory_df)} relevant files.")
    # print("Sample of inventory (first 5 and last 5 rows):")
    # print(inventory_df.head())
    # print("...")
    # print(inventory_df.tail())

def main():
    # The RAW_DATA_DIR_CONTAINING_CONTENT should now point directly to your 'HBD_Mixed_anions' folder
    content_root_path = RAW_DATA_DIR_CONTAINING_CONTENT
    inventory_full_path = INTERIM_DATA_DIR / INVENTORY_FILE_NAME

    create_file_inventory(content_root_path, inventory_full_path)

if __name__ == "__main__":
    main()