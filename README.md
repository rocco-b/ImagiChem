# ImagiChem
Rocco Buccheri¹ and Antonio Rescifina¹\
¹*University of Catania*

[![Python](https://img.shields.io/badge/Python-3.11.7-blue.svg)](https://www.python.org/)

## Setup and Installation
This repository contains the source code for the entire project in the 'ImagiChem code' folder.\
The installable file for Windows, and the executable file for Linux are in the ‘Releases’ section.

**Source code**\
To run from source code, clone the 'ImagiChem code' folder and install the pip packages:
```bash
pip install -r requirements.txt
```
Note: creating a virtual environment is recommended.

**Windows installer**\
Download the .msi file and run it. Windows will return an error message for unknown publisher. To install ImagiChem, follow these steps:
* Click ‘More info’
* Click ‘Run anyway’
The program will now be installed.

**Linux installer**\
If the executable file does not run, check that you have selected the following in ‘Permissions’:
* 'Allow executing file as program'

## Generation modes
1. **Hybrid — recommended**\
   Combines the library-based engine with the image-conditioned from-scratch backend, then merges unique SMILES.

2. **Library only**\
   Uses the ImagiChem v1.1 library/core-based engine.

3. **From-scratch only**\
   Uses the integrated rule-based from-scratch engine. The image profile controls ring count, aromatic/aliphatic balance, annulation/linker preference, and decoration patterns.

## Citation
If you find ImagiChem useful in your own research please cite:

**[Article approved title]**\
[Reference]
