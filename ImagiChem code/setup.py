from cx_Freeze import setup, Executable

build_options = {
    "packages": [
        "numpy",
        "cv2",
        "rdkit",
        "pubchempy",
        "PIL",
        "PyQt6"
    ],
    "excludes": []
}

exe = Executable(
    script="main.py",
    base="Win32GUI",             # no console window
    target_name="ImagiChem.exe",
    icon="icon.ico"
)

# Desktop shortcut
shortcut_table = [
    ("DesktopShortcut",
     "DesktopFolder",
     "ImagiChem",
     "TARGETDIR",
     "[TARGETDIR]ImagiChem.exe",
     None,
     None,
     None,
     None,
     None,
     None,
     "TARGETDIR")
]

msi_data = {"Shortcut": shortcut_table}
bdist_msi_options = {"data": msi_data}

setup(
    name="ImagiChem",
    version="1.0",
    description="Imagine Chemistry",
    options={
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options
    },
    executables=[exe]
)
