# src/plotting_functions.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import plotly.express as px # Only if you decide to put 3D Plotly code here instead of directly in dashboard

# Global constant for consistent x-axis in trend plots
ORDERED_X_VALUES_CONST = sorted([0, 1, 2, 3, 4, 5, 6])
DEFAULT_FIG_SIZE = (8, 5) # Default figure size for single plots
SMALL_FIG_SIZE = (7, 3.5) # For grid plots

def _format_property_name(prop_name: str) -> str:
    """Helper to make property names prettier for titles/labels."""
    return prop_name.replace('_hill_gpa', ' (Hill, GPa)') \
                    .replace('_gpa', ' (GPa)') \
                    .replace('_ev', ' (eV)') \
                    .replace('_ms', ' (m/s)') \
                    .replace('_k', ' (K)') \
                    .replace('is_mechanically_stable', 'Mech. Stable (1=Y)') \
                    .replace('_', ' ').title()

def plot_scalar_trends_grid(materials_df: pd.DataFrame, 
                            properties_to_plot: list = None, 
                            cols_per_row: int = 3,
                            main_title: str = "Trend of Material Properties vs. Anion Composition 'x'") -> plt.Figure | None:
    if materials_df.empty:
        print("PLOTTER: Materials DataFrame is empty for scalar trends grid.")
        return None

    if properties_to_plot is None:
        numeric_cols = materials_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cols_to_exclude = ['material_x_value', 'material_id']
        properties_to_plot = [col for col in numeric_cols if col not in cols_to_exclude]
    
    properties_available = [p for p in properties_to_plot if p in materials_df.columns and materials_df[p].notna().any()]
    if not properties_available:
        print("PLOTTER: No valid/available properties to plot in scalar trends grid.")
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "No data for trends grid.", ha='center'); return fig

    num_plots = len(properties_available)
    num_rows = (num_plots + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=cols_per_row, 
                             figsize=(cols_per_row * 5, num_rows * 3.8), sharex=True) # Adjusted fig height
    if num_plots == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, prop_name in enumerate(properties_available):
        ax = axes[i]
        sns.lineplot(data=materials_df, x='material_x_value', y=prop_name, marker='o', ax=ax, legend=False, errorbar=None)
        ax.set_title(_format_property_name(prop_name), fontsize=10)
        ax.set_ylabel("") 
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i >= num_plots - cols_per_row or num_rows == 1 : ax.set_xlabel("Chlorine Conc. 'x'", fontsize=9)
        ax.set_xticks(ORDERED_X_VALUES_CONST); ax.tick_params(axis='x', labelsize=8)

    for j in range(num_plots, len(axes)): fig.delaxes(axes[j])
    
    fig.suptitle(main_title, fontsize=16, y=1.02 if num_rows > 1 else 1.05, fontweight='bold')
    fig.tight_layout(pad=2.0, rect=[0, 0.03 if num_rows > 1 else 0.05, 1, 0.95 if num_rows > 1 else 0.92])
    return fig

def plot_correlation_heatmap(materials_df: pd.DataFrame,
                             title: str = "Correlation Matrix of Scalar Material Properties") -> plt.Figure | None:
    if materials_df.empty:
        print("PLOTTER: Materials DataFrame is empty for correlation heatmap."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "No data for heatmap.", ha='center'); return fig
        
    numeric_props = materials_df.select_dtypes(include=np.number).copy()
    cols_to_drop = ['material_id'] 
    for col in cols_to_drop:
        if col in numeric_props.columns: numeric_props.drop(columns=[col], inplace=True)

    if len(numeric_props.columns) <= 1:
        print("PLOTTER: Not enough numeric properties for correlation heatmap."); 
        fig, ax = plt.subplots(); ax.text(0.5,0.5, "Not enough data.", ha='center'); return fig
        
    correlation_matrix = numeric_props.corr()
    num_props_display = len(correlation_matrix.columns)
    figsize_w = max(8, num_props_display * 0.7)
    figsize_h = max(6, num_props_display * 0.6)
    
    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=.5, vmin=-1, vmax=1, center=0, 
                annot_kws={"size": max(5, 8 - num_props_display // 5)}, ax=ax) # Dynamic annot size
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelrotation=45, labelsize=max(6, 9 - num_props_display // 5))
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.tick_params(axis='y', labelrotation=0, labelsize=max(6, 9 - num_props_display // 5))
    fig.tight_layout()
    return fig

def plot_fundamental_band_gap_trend(materials_df: pd.DataFrame) -> plt.Figure | None:
    required_cols = ['material_x_value', 'band_character', 'direct_gap_ev', 'indirect_gap_ev']
    if materials_df.empty or not all(col in materials_df.columns for col in required_cols):
        print(f"PLOTTER: Missing one of {required_cols} in materials_df for band gap plot.")
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE); ax.set_title("Fundamental Band Gap (Data Missing)"); return fig

    temp_df = materials_df.copy()
    gaps = []
    for _, row in temp_df.iterrows():
        char, direct, indirect = row.get('band_character'), row.get('direct_gap_ev'), row.get('indirect_gap_ev')
        if char == 'Direct' and pd.notna(direct): gaps.append(direct)
        elif char == 'Indirect' and pd.notna(indirect): gaps.append(indirect)
        elif pd.notna(direct): gaps.append(direct) # Fallback
        elif pd.notna(indirect): gaps.append(indirect) # Fallback
        else: gaps.append(np.nan)
    temp_df['fundamental_gap_plot'] = gaps

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    direct_pts = temp_df[temp_df['band_character'] == 'Direct']
    if not direct_pts.empty: sns.scatterplot(data=direct_pts, x='material_x_value', y='fundamental_gap_plot', 
                                             marker='o', s=80, color='dodgerblue', label='Direct Gap', ax=ax, zorder=10)
    indirect_pts = temp_df[temp_df['band_character'] == 'Indirect']
    if not indirect_pts.empty: sns.scatterplot(data=indirect_pts, x='material_x_value', y='fundamental_gap_plot', 
                                               marker='X', s=100, color='orangered', label='Indirect Gap', ax=ax, zorder=10)
    
    if 'fundamental_gap_plot' in temp_df.columns and temp_df['fundamental_gap_plot'].notna().any():
        sns.lineplot(data=temp_df.dropna(subset=['fundamental_gap_plot']), 
                     x='material_x_value', y='fundamental_gap_plot', 
                     color='grey', ls='--', alpha=0.7, errorbar=None, ax=ax)

    ax.set_xlabel("Chlorine Conc. 'x'", fontsize=11); ax.set_ylabel("Fundamental Band Gap (eV)", fontsize=11)
    ax.set_title("Fundamental Band Gap vs. Anion Composition", fontsize=14, fontweight='bold')
    ax.set_xticks(ORDERED_X_VALUES_CONST); ax.tick_params(labelsize=9)
    if not direct_pts.empty or not indirect_pts.empty: ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.6); fig.tight_layout()
    return fig

def plot_fermi_energy_trend(materials_df: pd.DataFrame) -> plt.Figure | None:
    if materials_df.empty or 'fermi_energy_ev' not in materials_df.columns or 'material_x_value' not in materials_df.columns:
        print("PLOTTER: Missing data for Fermi energy plot.")
        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE); ax.set_title("Fermi Energy Trend (Data Missing)"); return fig

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    plot_data = materials_df.dropna(subset=['fermi_energy_ev'])
    if not plot_data.empty:
        sns.lineplot(data=plot_data, x='material_x_value', y='fermi_energy_ev', marker='o', color='darkgreen', errorbar=None, ax=ax)
        for _, row in plot_data.iterrows(): # Annotate points
            if pd.notna(row['fermi_energy_ev']):
                 ax.text(row['material_x_value'], row['fermi_energy_ev'], f" {row['fermi_energy_ev']:.3f}", ha='left', va='bottom', fontsize=7)
    else: ax.text(0.5, 0.5, "No Fermi energy data.", ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel("Chlorine Conc. 'x'", fontsize=11); ax.set_ylabel("Fermi Energy (eV)", fontsize=11)
    ax.set_title("Fermi Energy vs. Anion Composition", fontsize=14, fontweight='bold')
    ax.set_xticks(ORDERED_X_VALUES_CONST); ax.tick_params(labelsize=9)
    ax.grid(True, ls=':', alpha=0.6); ax.axhline(0, color='black', ls='-', lw=0.5, alpha=0.5)
    fig.tight_layout()
    return fig

# --- DETAILED PLOTS FOR A SINGLE SELECTED MATERIAL ---
def plot_band_structure_single_material(bands_df: pd.DataFrame, klabels_df: pd.DataFrame, 
                                        material_info_row: pd.Series, x_val: int, 
                                        calc_type: str = 'GGA') -> plt.Figure | None:
    if bands_df is None or bands_df.empty: return None
    fig, ax = plt.subplots(figsize=(7, 5.5)) # Adjusted size
    fermi, gap, formula = 0.0, 2.0, f"x={x_val}" # Defaults
    if not material_info_row.empty:
        formula = material_info_row.get('material_formula', formula)
        fermi = material_info_row.get('fermi_energy_ev', 0.0) if pd.notna(material_info_row.get('fermi_energy_ev')) else 0.0
        gap = material_info_row.get('direct_gap_ev', 2.0) if pd.notna(material_info_row.get('direct_gap_ev')) else 2.0
        
    for band_idx in sorted(bands_df['band_index'].unique()):
        data = bands_df[bands_df['band_index'] == band_idx]
        ax.plot(data['k_distance'], data['energy_ev'] - fermi, color='dodgerblue', lw=0.85)
    
    ax.axhline(0, color='red', ls='--', lw=1, label='E$_F$ (0 eV)')
    if klabels_df is not None and not klabels_df.empty and 'k_coord' in klabels_df.columns and 'k_label' in klabels_df.columns:
        unique_k_coords, unique_labels = [], []
        for k_c, grp in klabels_df.groupby('k_coord'):
            unique_k_coords.append(k_c); unique_labels.append(" | ".join(grp['k_label'].unique()))
            ax.axvline(x=k_c, color='grey', ls=':', lw=0.6, alpha=0.7)
        ax.set_xticks(unique_k_coords); ax.set_xticklabels(unique_labels, fontsize=8)
    
    ax.set_xlabel("High Symmetry K-Path", fontsize=10)
    ax.set_ylabel(f"Energy {'(E - E$_F$)' if pd.notna(fermi) and fermi != 0.0 else ''} (eV)", fontsize=10)
    ax.set_title(f"Band Structure: {formula} (x={x_val}) - {calc_type}", fontsize=12)
    ymin, ymax = -max(2.0, gap * 1.5), max(2.0, gap * 1.5)
    ax.set_ylim(ymin, ymax); ax.set_xlim(bands_df['k_distance'].min(), bands_df['k_distance'].max())
    ax.legend(fontsize=8); ax.grid(True, axis='y', ls=':', alpha=0.5); fig.tight_layout()
    return fig

def plot_tdos_single_material(dos_df: pd.DataFrame, material_info_row: pd.Series, x_val: int) -> plt.Figure | None:
    if dos_df is None or dos_df.empty or 'tdos' not in dos_df.columns or 'Energy_eV' not in dos_df.columns: return None
    fig, ax = plt.subplots(figsize=(7, 4.5)) # Adjusted size
    fermi, formula = 0.0, f"x={x_val}"
    if not material_info_row.empty:
        formula = material_info_row.get('material_formula', formula)
        fermi = material_info_row.get('fermi_energy_ev', 0.0) if pd.notna(material_info_row.get('fermi_energy_ev')) else 0.0
        
    shifted_e = dos_df['Energy_eV'] - fermi
    ax.plot(shifted_e, dos_df['tdos'], label='Total DOS', lw=1.5, color='navy')
    # Add spin plotting if available
    if 'dos_spin_up' in dos_df.columns and dos_df['dos_spin_up'].notna().any():
        ax.plot(shifted_e, dos_df['dos_spin_up'], label='Spin Up', lw=1, alpha=0.7, color='green')
    if 'dos_spin_down' in dos_df.columns and dos_df['dos_spin_down'].notna().any():
        ax.plot(shifted_e, dos_df['dos_spin_down'], label='Spin Down', lw=1, alpha=0.7, color='purple')
        
    ax.axvline(0, color='black', ls='--', lw=1, label='E$_F$ (0 eV)')
    ax.set_xlabel(f"Energy {'(E - E$_F$)' if pd.notna(fermi) and fermi != 0.0 else ''} (eV)", fontsize=10)
    ax.set_ylabel("DOS (States/eV/Cell)", fontsize=10) # Shortened unit
    ax.set_title(f"Total DOS for {formula} (x={x_val})", fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.6); ax.set_xlim(-6, 6); ax.set_ylim(bottom=0) # Adjusted xlim
    fig.tight_layout()
    return fig

def plot_optical_spectra_subplots_material(optical_df_for_selected_x: pd.DataFrame, 
                                         material_info_row: pd.Series, x_val: int) -> plt.Figure | None:
    """Generates a figure with subplots for key optical spectra for a SINGLE material."""
    if optical_df_for_selected_x is None or optical_df_for_selected_x.empty: return None

    spectra_config = {
        'ABSORPTION': ('Absorption Coeff. (arb. units)', 'val_avg'), 
        'REFLECTIVITY': ('Reflectivity', 'val_avg'),
        'REFRACTIVE_N': ('Refractive', 'val_avg'),
    }
    fig, axes = plt.subplots(nrows=len(spectra_config), ncols=1, 
                             figsize=(8, len(spectra_config) * 3.5), sharex=True) # Adjusted fig height
    if len(spectra_config) == 1: axes = np.array([axes])

    formula = material_info_row.get('material_formula', f"x={x_val}")
    
    for i, (spec_type, (ylabel, y_col_default)) in enumerate(spectra_config.items()):
        ax = axes[i]
        data_for_spectrum = optical_df_for_selected_x[optical_df_for_selected_x['spectrum_type'] == spec_type]
        
        y_col_to_plot = y_col_default
        if y_col_default not in data_for_spectrum.columns or not data_for_spectrum[y_col_default].notna().any():
            if 'val_xx' in data_for_spectrum.columns and data_for_spectrum['val_xx'].notna().any():
                y_col_to_plot = 'val_xx' # Fallback to xx component
            else:
                ax.text(0.5,0.5, "No data for y-axis.", ha='center'); ax.set_title(spec_type); continue

        if not data_for_spectrum.empty and y_col_to_plot in data_for_spectrum.columns:
            ax.plot(data_for_spectrum['photon_energy_ev'], data_for_spectrum[y_col_to_plot], lw=1.5)
        else:
            ax.text(0.5,0.5, "Data not available.", ha='center')
        
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, ls=':', alpha=0.6)
        ax.set_title(spec_type.replace('_', ' ').title(), fontsize=10, loc='left')
        if "absorption" in ylabel.lower() or "reflectivity" in ylabel.lower(): ax.set_ylim(bottom=0)
        ax.set_xlim(0, 6) # Common optical range

    if len(spectra_config) > 0:
        axes[-1].set_xlabel("Photon Energy (eV)", fontsize=10)
        fig.suptitle(f"Optical Spectra for {formula} (x={x_val})", fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig