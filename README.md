# DFT Studies of the Role of Anion Variation in Physical Properties of Cs2NaTlBr6-xClx (x = 0, 1, 2, 3, 4, 5, and 6) Mixed Halide Double Perovskites for Optoelectronics

Dataset DOI: [10.5061/dryad.8gtht770d](10.5061/dryad.8gtht770d)

## Description of the data and file structure

We conducted the theoretical and simulation study of mixed halide double perovskites Cs2NaTlBr6-xClx based on the density functional theory. We investigated the structural, mechanical, and thermodynamic stability of the considered materials and electronic and optical properties to determine the material's potential as a candidate for optoelectronics. In HBD_Mixed_anions.zip, the different data types folder is for two pure double perovskites, such as Cs₂NaTlCl₆ & Cs₂NaTlBr₆. Further, the Cl2Br4, Cl4Br2 (x = 2, 4) in ClxBr6-xData33 and ClBr5, Cl3Br3, Cl5Br (x= 1, 3, 5) in ClxBr1-xpart4 files are arranged for Cs2NaTlBr6-xClx mixed halide double perovskites. In each of the folders, readers will find the dataset for the calculation of structural optimization or relaxation files (SR), self-consistent calculations (STR), structural properties (STRUC), band structure (Band), hybrid band structure (HSE), density of states (DOS), effective mass calculation (eff_M), mechanical properties (Elastic), and optical properties (OPTIC). If someone wants to perform the calculations, they need to have experience at least in VASP or similar tools for first-principles calculations in computational materials science.

### Files and variables

#### File: HBD\_Mixed\_anions.zip

**Description:** In this submitted file, we included the data files to carry out the calculations using the Vienna ab-initio simulation package (VASP). We divided all the dataset in a specific folder name so that any one can track all the data.

i. The four input files are always needed to perform a calculation in VASP. These files are POSCAR, POTCAR, INCAR, and KPOINTS.

ii. OUTCAR stores all the information regarding the calculations steps.

iii. The information about the total density of states and partial density of states in tdos.dat and PDOS_USER1.dat, PDOS_USER2.dat, PDOS_USER3.dat, PDOS_USER4.dat, PDOS_USER5.dat files, respectively within the DOS folder.

iv. The bandgap related information will be found in BAND_GAP and for the illustration of the bandstructure, KLABELS, HIGH_SYMMETRY_POINTS, BAND.dat, REFORMATTED_BAND.dat are necessary files.

v. We have used VPKIT.in file for calculations of the structural properties and effective mass of electron using different parameter settings.

vi. In OPTIC folder, all the parameters such as real and imaginary dielectric function in real.in and IMAG.in, absorption coefficient in ABSORPTION.dat, extinction coefficient in EXTINCTION.dat, energy loss function in ENERGY_LOSSSPECTRUM.dat, reflectivity of the material in REFLECTIVITY.dat, and refractive index in REFRACTIVE.dat files are arranged with respect to energy (eV).

## Code/software

To visualize the unit cell of the compound (POSCAR), we used Vesta software. Further, all the calculations can be done by the VASP simulation package. Matplotlib is used to visualize all the graphs in this research. The .dat file store the output data for the each calculation in respective folders with proper nametag with them.

## Access information

Other publicly accessible locations of the data:

* Not Applicable

Data was derived from the following sources:

* Vasp, Vaspkit, Vesta, Matplotlib

