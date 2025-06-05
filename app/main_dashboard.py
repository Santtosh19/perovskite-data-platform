# app/main_dashboard.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import joblib # For loading saved ML models
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib 
import sys

# --- Add 'src' to sys.path to find plotting_functions ---
# This assumes main_dashboard.py is in app/ and plotting_functions.py is in src/
current_script_dir = pathlib.Path(__file__).resolve().parent
project_root = current_script_dir.parent 
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    # print(f"DASHBOARD: Added to sys.path: {src_path}")

try:
    from plotting_functions import ( # Ensure this matches your filename if it's not exactly plotting_functions.py
        plot_scalar_trends_grid, plot_correlation_heatmap, 
        plot_fundamental_band_gap_trend, plot_fermi_energy_trend,
        plot_band_structure_single_material, 
        plot_tdos_single_material,    
        plot_optical_spectra_subplots_material,
        _format_property_name
    )
except ModuleNotFoundError:
    st.error("CRITICAL: Could not import 'plotting_functions.py'. Ensure it exists in the 'src' directory and 'src' is in sys.path.")
    st.stop()


# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Perovskite Insights Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Setup & Constants ---
DB_FILE_NAME = "perovskite_platform.sqlite"
DB_PATH = str(project_root / "data" / "processed" / DB_FILE_NAME)
MODELS_DIR = str(project_root / 'src'/'surrogate_models_trained') # Models directory at project root

ORDERED_X_VALUES = sorted([0, 1, 2, 3, 4, 5, 6])
PROPERTIES_MODELED_FOR_SURROGATE = ['direct_gap_ev', 'bulk_modulus_k_hill_gpa', 'debye_temperature_k']

# --- Data Loading Functions with Streamlit Caching ---
@st.cache_data
def load_summary_data(db_path_str):
    # print("CACHE MISS: Loading Materials Summary from DB")
    conn = None
    try:
        conn = sqlite3.connect(db_path_str)
        df = pd.read_sql_query("SELECT * FROM Materials ORDER BY material_x_value", conn)
        for col in ['is_mechanically_stable', 'criteria_c11_gt_c12_met', 'criteria_c11_add_2c12_gt_0_met', 'criteria_c44_gt_0_met']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: True if x == 1 else (False if x == 0 else None)).astype('object')
        return df
    except Exception as e:
        st.error(f"Error loading Materials summary from '{db_path_str}': {e}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()

@st.cache_data
def load_spectral_or_band_data(_db_path_for_cache_key, table_name, x_value=None, spectrum_type_filter=None, calc_type_filter=None):
    db_path_str = _db_path_for_cache_key
    # print(f"CACHE MISS: Loading '{table_name}' for x={x_value}, type='{spectrum_type_filter}', calc='{calc_type_filter}'")
    conn = None
    
    # Check if table exists first
    try:
        conn_check = sqlite3.connect(db_path_str)
        cursor_check = conn_check.cursor()
        cursor_check.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor_check.fetchone() is None:
            st.warning(f"Table '{table_name}' not found in database '{db_path_str}'.")
            if conn_check: conn_check.close()
            return pd.DataFrame()
        if conn_check: conn_check.close()
    except Exception as e_check:
        st.error(f"Error checking if table '{table_name}' exists: {e_check}")
        return pd.DataFrame()

    base_query = f"SELECT * FROM \"{table_name}\""
    conditions = []
    if x_value is not None: conditions.append(f"material_x_value = {int(x_value)}") # Ensure x_value is int

    # To check if spectrum_type/calc_type columns exist, would need a quick PRAGMA query
    # For simplicity, we'll assume if the filter is provided, the column should exist
    if spectrum_type_filter is not None: conditions.append(f"spectrum_type = '{spectrum_type_filter}'")
    if calc_type_filter is not None: conditions.append(f"calc_type = '{calc_type_filter}'")
        
    if conditions: base_query += " WHERE " + " AND ".join(conditions)
    
    # Add ordering (ensure column exists before ordering by it)
    # Example ordering, you may want to fetch schema to confirm column existence
    if table_name == "BandStructureData": base_query += " ORDER BY k_distance, band_index"
    elif table_name == "KLabelsData": base_query += " ORDER BY \"K-Coordinate in band-structure plots\""
    elif table_name == "OpticalSpectraData": base_query += " ORDER BY photon_energy_ev"
    elif table_name == "DensityOfStatesData": base_query += " ORDER BY Energy_eV"

    try:
        conn = sqlite3.connect(db_path_str)
        df = pd.read_sql_query(base_query, conn)
        return df
    except Exception as e:
        st.error(f"Error loading data from '{table_name}': {e}. Query attempted: {base_query}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()

@st.cache_resource
def load_ml_model(model_filename):
    # print(f"CACHE MISS: Loading ML model from {model_filename}")
    model_path = os.path.join(MODELS_DIR, model_filename)
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model {model_path}: {e}")
            return None
    # st.warning(f"Model file not found: {model_path}") # Can be noisy if models aren't trained yet
    return None

# --- Load initial data and models ---
materials_df = load_summary_data(DB_PATH)
if materials_df.empty:
    st.error("CRITICAL: Materials summary data not loaded. Dashboard functionality will be limited.")
    # st.stop() # Decide if you want to halt execution entirely

os.makedirs(MODELS_DIR, exist_ok=True) 
surrogate_models = {}
for prop_name in PROPERTIES_MODELED_FOR_SURROGATE:
    surrogate_models[f'{prop_name}_poly'] = load_ml_model(f'poly_model_{prop_name}.pkl')
    surrogate_models[f'{prop_name}_gpr'] = load_ml_model(f'gpr_model_{prop_name}.pkl')

# --- App Title & Introduction ---
st.title("⚛️ Perovskite Material Insights Platform")
st.markdown("""
Explore DFT-calculated properties of Cs₂NaTlBr₆₋ₓClₓ mixed halide double perovskites.
This platform allows visualization of trends, detailed property exploration for each composition,
and predictive interpolation using surrogate models.
""")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("User Selections")
selected_x_discrete = "N/A" # Default
if not materials_df.empty and 'material_x_value' in materials_df.columns:
    material_labels = {
        x: f"x={x} ({materials_df.loc[materials_df['material_x_value'] == x, 'material_formula'].iloc[0]})"
        for x in ORDERED_X_VALUES if x in materials_df['material_x_value'].unique()
    }
    # Ensure options are derived from what's actually in materials_df if it might be incomplete
    x_options_available = [x for x in ORDERED_X_VALUES if x in materials_df['material_x_value'].unique()]
    if x_options_available:
        selected_x_discrete = st.sidebar.selectbox(
            "Select Material Composition:",
            options=x_options_available,
            format_func=lambda x: material_labels.get(x, f"x={x}")
        )
    else:
        st.sidebar.text("No material compositions available for selection.")
else:
    st.sidebar.selectbox("Select Material Composition (x value):", options=["Data not loaded"])

selected_x_continuous = st.sidebar.number_input(
    "Predict for 'x' (0.0-6.0):", min_value=0.0, max_value=6.0, value=2.5, step=0.1, format="%.2f"
)
st.sidebar.markdown("---")

if selected_x_discrete != "N/A" and not materials_df.empty:
    material_info_row_series = materials_df[materials_df['material_x_value'] == selected_x_discrete].iloc[0] if not materials_df[materials_df['material_x_value'] == selected_x_discrete].empty else pd.Series()
    if not material_info_row_series.empty:
        st.sidebar.markdown(f"**Summary for {material_info_row_series.get('material_formula','')} (x={selected_x_discrete}):**")
        key_sidebar_props = {'direct_gap_ev': 'Direct Gap (eV)', 'bulk_modulus_k_hill_gpa': 'Bulk Modulus (GPa)', 'debye_temperature_k': 'Debye Temperature (K)', 'is_mechanically_stable': 'Mechanically Stable'}
        for prop_key, display_label in key_sidebar_props.items():
            if prop_key in material_info_row_series.index: # Check if key exists
                val = material_info_row_series[prop_key]
                display_val = "N/A"
                if pd.notna(val):
                    if isinstance(val, float): display_val = f"{val:.3f}"
                    elif prop_key == 'is_mechanically_stable': display_val = "Yes" if val is True else "No" # Bool directly
                    else: display_val = str(val)
                st.sidebar.text(f"{display_label}: {display_val}")
st.sidebar.markdown("---")

# --- Main Area Tabs ---
tab_titles = ["📈 Scalar Trends", "🔬 Material Detail", "🔮 Predictions", "⚖️ Trade-offs", "ℹ️ About"]
tabs = st.tabs(tab_titles)

with tabs[0]: # Scalar Trends
    st.header("Trends of Scalar Properties")
    if not materials_df.empty:
        st.subheader("Fundamental Band Gap")
        fig_gap = plot_fundamental_band_gap_trend(materials_df)
        if fig_gap: st.pyplot(fig_gap)
        st.markdown("---")
        st.subheader("Fermi Energy")
        fig_fermi = plot_fermi_energy_trend(materials_df)
        if fig_fermi: st.pyplot(fig_fermi)
        st.markdown("---")
        st.subheader("Other Scalar Property Trends Grid")
        # Define a subset of interesting properties for the grid
        grid_props_subset = ['bulk_modulus_k_hill_gpa', 'debye_temperature_k', 'pugh_ratio', 'is_mechanically_stable', 
                             'poissons_ratio_v_hill', 'youngs_modulus_e_hill_gpa']
        fig_grid = plot_scalar_trends_grid(materials_df, properties_to_plot=grid_props_subset, cols_per_row=2)
        if fig_grid: st.pyplot(fig_grid)
        st.markdown("---")
        st.subheader("Correlation Matrix")
        fig_corr = plot_correlation_heatmap(materials_df)
        if fig_corr: st.pyplot(fig_corr)
    else: st.warning("Materials summary data not available.")

with tabs[1]: # Material Detail
    st.header(f"Detailed Plots for Material x = {selected_x_discrete}")
    if selected_x_discrete != "N/A" and not materials_df.empty:
        current_material_info_row = materials_df[materials_df['material_x_value'] == selected_x_discrete].iloc[0] if not materials_df[materials_df['material_x_value'] == selected_x_discrete].empty else pd.Series()

        if not current_material_info_row.empty:
            st.subheader(f"Band Structure ({current_material_info_row.get('material_formula', '')})")
            bs_df = load_spectral_or_band_data(DB_PATH, "BandStructureData", x_value=selected_x_discrete, calc_type_filter='GGA')
            kl_df = load_spectral_or_band_data(DB_PATH, "KLabelsData", x_value=selected_x_discrete, calc_type_filter='GGA')
            if not kl_df.empty and "K_Label" in kl_df.columns and "K-Coordinate in band-structure plots" in kl_df.columns: # Standardize KLabel col names if needed
                kl_df = kl_df.rename(columns={"K_Label": "k_label", "K-Coordinate in band-structure plots": "k_coord"}, errors='ignore')

            fig_bs = plot_band_structure_single_material(bs_df, kl_df, current_material_info_row, selected_x_discrete, calc_type='GGA')
            if fig_bs: st.pyplot(fig_bs)
            else: st.write("Band structure plot could not be generated.")

            st.markdown("---")
            st.subheader(f"Total Density of States (TDOS) for {current_material_info_row.get('material_formula', '')}")
            dos_df = load_spectral_or_band_data(_db_path_for_cache_key=DB_PATH, table_name="DensityOfStatesData", x_value=selected_x_discrete)
            fig_dos = plot_tdos_single_material(dos_df, current_material_info_row, selected_x_discrete)
            if fig_dos: st.pyplot(fig_dos)
            else: st.write("TDOS plot could not be generated.")
            
            st.markdown("---")
            st.subheader(f"Optical Spectra for {current_material_info_row.get('material_formula', '')}")
            # Pass DB_PATH for this plotter as it handles its own data loading internally for subplots
            all_optics_for_this_x = load_spectral_or_band_data(
            _db_path_for_cache_key=DB_PATH,  # Pass DB_PATH as the cache key helper
            table_name="OpticalSpectraData", 
            x_value=selected_x_discrete)
            if not all_optics_for_this_x.empty:
            # STEP 2: Now pass the loaded DataFrame to your plotting function
                fig_optics = plot_optical_spectra_subplots_material(
                optical_df_for_selected_x=all_optics_for_this_x, # <<< PASS THE DATAFRAME HERE
                material_info_row=current_material_info_row, 
                x_val=selected_x_discrete # Pass x_val again for titles etc. in plotter
            )
            if fig_optics:
                st.pyplot(fig_optics)
            else:
                # This branch might be hit if plot_optical_spectra_subplots_material returns None
                # (e.g., if after filtering within the plotter, no actual data for subplots was found)
                st.write(f"Optical spectra plot could not be generated for x={selected_x_discrete} (plot function returned None).")
        else:
            st.write(f"No optical data found in the database for x={selected_x_discrete} to generate plots.")


with tabs[2]: # Predictions
    st.header(f"Surrogate Model Predictions for x = {selected_x_continuous:.2f}")
    st.markdown("Estimations based on models trained on 7 DFT data points for selected properties.")
    X_predict_val_np = np.array([[selected_x_continuous]])

    for prop_name in PROPERTIES_MODELED_FOR_SURROGATE:
        prop_display_name = prop_name.replace('_',' ').replace('ev','(eV)').replace('gpa','(GPa)').replace('k','(K)').title()
        st.markdown(f"#### Predicted: {prop_display_name}")
        
        col1, col2 = st.columns(2)
        poly_model = surrogate_models.get(f'{prop_name}_poly')
        if poly_model:
            try:
                pred_poly = poly_model.predict(X_predict_val_np)[0]
                col1.metric(label=f"Polynomial Fit", value=f"{pred_poly:.4f}")
            except Exception as e: col1.error(f"Poly Model Error: {e}")
        else: col1.warning("Poly model not loaded.")
            
        gpr_model = surrogate_models.get(f'{prop_name}_gpr')
        if gpr_model:
            try:
                pred_gpr_mean, pred_gpr_std = gpr_model.predict(X_predict_val_np, return_std=True)
                col2.metric(label=f"GPR Fit", value=f"{pred_gpr_mean[0]:.4f}", delta=f"± {(1.96 * pred_gpr_std[0]):.4f} (95% CI)", delta_color="off")
            except Exception as e: col2.error(f"GPR Model Error: {e}")
        else: col2.warning("GPR model not loaded.")
        # st.markdown("---") # Removed for tighter look

with tabs[3]: # Trade-offs
    st.header("Multi-Property Trade-off Explorer")
    if not materials_df.empty:
        st.markdown("Select three scalar properties for the 3D scatter plot:")
        numeric_scalars = sorted([col for col in materials_df.columns if pd.api.types.is_numeric_dtype(materials_df[col]) and col not in ['material_x_value', 'material_id']])
        
        if len(numeric_scalars) >= 3:
            default_x = 'direct_gap_ev' if 'direct_gap_ev' in numeric_scalars else numeric_scalars[0]
            default_y = 'bulk_modulus_k_hill_gpa' if 'bulk_modulus_k_hill_gpa' in numeric_scalars else numeric_scalars[1]
            default_z = 'debye_temperature_k' if 'debye_temperature_k' in numeric_scalars else numeric_scalars[2]

            sel_x_3d = st.selectbox("X-axis:", numeric_scalars, index=numeric_scalars.index(default_x))
            options_y_3d = [col for col in numeric_scalars if col != sel_x_3d]
            sel_y_3d = st.selectbox("Y-axis:", options_y_3d, index=options_y_3d.index(default_y) if default_y in options_y_3d else 0)
            options_z_3d = [col for col in options_y_3d if col != sel_y_3d]
            sel_z_3d = st.selectbox("Z-axis:", options_z_3d, index=options_z_3d.index(default_z) if default_z in options_z_3d else 0)

            cols_for_3d = [sel_x_3d, sel_y_3d, sel_z_3d]
            plot_df_3d = materials_df[['material_x_value', 'material_formula'] + cols_for_3d].copy().dropna(subset=cols_for_3d)

            if len(plot_df_3d) >= 2:
                fig_3d = px.scatter_3d(plot_df_3d, x=sel_x_3d, y=sel_y_3d, z=sel_z_3d,
                                    color='material_x_value', hover_name='material_formula',
                                    color_continuous_scale=px.colors.sequential.Plasma,
                                    title=f"{_format_property_name(sel_x_3d)} vs {_format_property_name(sel_y_3d)} vs {_format_property_name(sel_z_3d)}")
                fig_3d.update_layout(margin=dict(l=0,r=0,b=0,t=50), legend_title_text="Cl Conc. 'x'",
                                     scene=dict(xaxis_title=_format_property_name(sel_x_3d),
                                                yaxis_title=_format_property_name(sel_y_3d),
                                                zaxis_title=_format_property_name(sel_z_3d)))
                st.plotly_chart(fig_3d, use_container_width=True)
            else: st.warning("Not enough data for 3D plot with selected properties after dropping NaNs.")
        else: st.warning("Not enough numeric properties for 3D plotting.")
    else: st.warning("Materials data not available.")

with tabs[4]: # About
    st.header("About This Platform")
    # (Your existing "About" markdown content - make sure to include your name and GitHub link)
    st.markdown("""
    This interactive platform visualizes and explores DFT-calculated properties of Cs₂NaTlBr₆₋ₓClₓ 
    mixed halide double perovskites. The primary data is sourced from the study by:
    *Hasan, Mohammed Mehedi; Hasan, Nazmul; Kabir, Alamgir (2025). DFT studies of the role of anion variation in physical properties of Cs2NaTlBr6-xClx (x = 0, 1, 2, 3, 4, 5, and 6) mixed halide double perovskites for optoelectronics. Dryad, Dataset, https://doi.org/10.5061/dryad.8gtht770d*

    **Key Project Components:**
    - Data Ingestion & Parsing of raw DFT outputs.
    - SQLite Database for structured data storage.
    - Exploratory Data Analysis & Visualization (Pandas, Matplotlib, Seaborn, Plotly).
    - Surrogate Modeling (Scikit-learn: Polynomial & Gaussian Process Regression) for property interpolation.
    - This Interactive Dashboard built with Streamlit.

    This project demonstrates end-to-end data processing, analysis, modeling, and presentation skills.

    **Developed by:** [SANTTOSH G MUNIYANDY]
    **GitHub Repository:** [https://github.com/Santtosh19]
    """)