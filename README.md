# ⚛️ Perovskite Material Insights Platform  perovskite-data-platform

**Live Interactive Dashboard: https://santtosh19-perovskite-data-platform-appmain-dashboard-ujrjl4.streamlit.app/** 

## 📖 Overview

This project presents an interactive platform for exploring the calculated properties of Cs₂NaTlBr₆₋ₓClₓ mixed halide double perovskites. Utilizing Density Functional Theory (DFT) simulation output data from the study by Hasan et al. (2025), this platform enables users to:

*   Visualize how fundamental material properties (electronic, optical, mechanical) evolve with varying anion (Bromine/Chlorine) compositions.
*   Interactively explore detailed electronic band structures, density of states, and optical spectra for each of the 7 studied compositions (x=0 to x=6).
*   Identify potential design trade-offs between multiple desirable material characteristics.
*   Obtain rapid estimations of key properties for untested intermediate compositions using trained surrogate models.

The primary goal is to make complex scientific data more accessible and actionable, facilitating a deeper understanding of these promising lead-free perovskite materials for potential optoelectronic applications.

## ✨ Key Achievements & Features

This project goes beyond simple data visualization by implementing an end-to-end data pipeline and a predictive component:

1.  **Unified & Queryable Data Platform:**
    *   **Challenge Addressed:** Raw DFT outputs are often in numerous, disparate file formats, making holistic analysis difficult.
    *   **Solution:** Developed a robust Python-based data engineering pipeline to automatically parse, validate, and structure the diverse DFT output files into a centralized SQLite database. This involved creating custom parsers for specific scientific data formats.
    *   **Benefit:** Enabled systematic querying, comprehensive property exploration, and served as the foundation for all subsequent advanced analyses.

2.  **Predictive Surrogate Modeling:**
    *   **Challenge Addressed:** DFT calculations are computationally expensive, limiting the number of compositions that can be simulated.
    *   **Solution:** Trained surrogate models (Polynomial Regression and Gaussian Process Regression using Scikit-learn) on the 7 available DFT data points to predict key material properties (Direct Bandgap, Bulk Modulus, Debye Temperature) as a continuous function of anion concentration 'x'.
    *   **Benefit:** Provides a novel capability for rapid property estimation for untested intermediate compositions, potentially guiding and accelerating future research by identifying promising 'x' values. GPR models also offer valuable uncertainty quantification for these predictions.

3.  **Interactive Multi-Property Trade-Off Explorer:**
    *   **Challenge Addressed:** Materials design often involves balancing multiple, sometimes competing, desirable properties.
    *   **Solution:** Implemented interactive 3D visualizations (using Plotly) allowing users to simultaneously explore relationships between three user-selected scalar properties, colored by composition.
    *   **Benefit:** Offers new visual insights into material design trade-offs and helps identify "Pareto-like" optimal compositions.

4.  **Comprehensive Interactive Dashboard:**
    *   **Challenge Addressed:** Making complex scientific data easily explorable by a wider audience.
    *   **Solution:** Developed a user-friendly web application using Streamlit.
    *   **Benefit:** Transforms static research data into a dynamic tool, enhancing data accessibility and enabling user-driven exploration of trends, detailed material properties, and model predictions.

## 🛠️ Tech Stack & Libraries

*   **Language:** Python 3.x
*   **Data Manipulation & Analysis:** Pandas, NumPy
*   **Database:** SQLite3 (managed via Python)
*   **Data Parsing:** Custom Python scripts (using `os`, `pathlib`, `re`)
*   **Machine Learning (Surrogate Models):** Scikit-learn (`PolynomialFeatures`, `LinearRegression`, `GaussianProcessRegressor`)
*   **Model Persistence:** Joblib (`.pkl` files)
*   **Data Visualization:** Matplotlib, Seaborn, Plotly (for interactive 3D plots)
*   **Web Application Framework:** Streamlit
*   **Version Control:** Git & GitHub


## 🚀 Getting Started / How to Run Locally

### Prerequisites
*   Python 3.9+
*   Git

### Setup & Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Santtosh19/perovskite-data-platform.git
    cd perovskite-data-platform
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Data Acquisition - if `data/raw/HBD_Mixed_anions/` is not included in repo)**
    *   Download the dataset `HBD_Mixed_anions.zip` from Dryad: [https://doi.org/10.5061/dryad.8gtht770d](https://doi.org/10.5061/dryad.8gtht770d)
    *   Unzip it and place the contents (the `HBD_Mixed_anions` folder itself containing `Cs2NaTlCl6`, `Cs2NaTlBr6`, etc.) into the `data/raw/` directory of this project.

### Running the Data Pipeline & Training Models
To generate the SQLite database and train the surrogate models from scratch:

1.  **Run the file inventory script:**
    ```bash
    python ingestion_scripts/01_unzip_and_inventory.py
    ```
2.  **Run the data parsing and CSV generation script:**
    ```bash
    python ingestion_scripts/02_extract_and_structure_data.py
    ```
3.  **Build the SQLite database:**
    ```bash
    python ingestion_scripts/03_build_database.py
    ```
4.  **Train surrogate models (generates `.pkl` files in `models/`):**
    *   Open and run the Jupyter Notebook: `notebooks/02_Surrogate_Modeling.ipynb`

*(Note: If the pre-populated `data/processed/perovskite_platform.sqlite` and pre-trained `models/*.pkl` files are included in the repository, you can skip the pipeline and training steps to directly run the dashboard).*

### Running the Streamlit Dashboard
Once the database and models are ready (either generated or included):
```bash
streamlit run app/main_dashboard.py
```

## 🔮 Future Work
Incorporate HSE-level calculations for all properties across all compositions if data becomes available.
Extend surrogate models to predict a wider range of properties and explore multi-target regression.
Integrate experimental data for these compounds (if published) to validate DFT and model predictions.
Develop more sophisticated tools for identifying optimal materials based on user-defined multi-objective criteria.

## 👨‍💻 About the Author
Santtosh G. Muniyandy
GitHub: https://github.com/Santtosh19
