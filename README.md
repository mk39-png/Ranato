# Ranato 

Blender add-on showcasing a practical application of PYAC for technical artists.


## Dependencies 
* CMake 3.29 or any version that supports CMake 2.8 at the earliest to compile Cholespy
* Blender 4.5 or newer 
* Python 3.11.9 (or whatever version Blender utilizes)
* Visual Studio Community 2026 or 2022 OR Visual Studio Build Tools 2026 (we just need the MSVC compilation toolchain)
    * Install "C++ Desktop Development"   

## Installation

Be sure to replace [Repos] with the location of where you store your repositories.

### Clones PYAC Python library with its dependencies (i.e. Cholespy external library)
git clone --recursive https://github.com/mk39-png/py-algebraic-contours.git

### OPTIONAL: install requirements into virtual environment of Ranato for typing and context
pip install -e [directory to py-algebraic-contours]
pip install -r requirements.txt [within the workspace folder of the Ranato addon]

### Install PYAC wheel into Ranato
pip wheel --no-cache-dir D:\[Repos]\py-algebraic-contours\ -w D:\[Repos]\Ranato\ranato\wheels\ 

### Install Cholespy wheel (to overwrite the Cholespy install from PyPi)
pip wheel --no-cache-dir D:\[Repos]\py-algebraic-contours\external\cholespy\ -w D:\[Repos]\Ranato\ranato\wheels\


## Run (Development) 

For development, Ranato can be ran in Blender using the Blender Development add-on in Visual Studio Code.
https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development


## Troubleshooting
Below are common errors one may find when installing Ranato and its dependencies such as PYAC or Cholespy.

### ISSUE
“Compatibility with CMake < XX has been removed from CMake”

#### REASON
Do NOT compile Cholespy with too new of a CMAKE version (e.g. 4.2.1) since it does not support CMAKE 2.8, 3.1, 3.5, and 3.10 that Cholespy uses.

#### FIX 
CMAKE in System Environment Variables needs to be at the TOP in User Variables for [local system username] so that it is detected and used first rather than the CMAKE included with Visual Studio’s MSVC toolchain (i.e. include C:\Program Files\CMake\bin at the top of the environment variables)


### ISSUE 
“CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage” “CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage”

#### REASON
For people with Strawberry install, be sure to switch away from that (since Cholespy will automatically compile for Ninja) and uninstall it. Or, find a way to switch toolchains to MSVC. 

##### FIX
Uninstall Strawberry or attempt to swap toolchains to MSVC. The terminal should say something like “Building for Visual Studio 18 2026” when running “pip install ./cholespy”


### ISSUE
Code changes to PYAC are not being reflected in Ranato.

#### REASON
Blender refers to the old wheels generated from your code. An ideal version would have a script that rebuilds the wheel and reinstalls Ranato into Blender with the new, updated code.

#### FIX
To run an updated version of Ranato, be sure to uninstall the Ranato add-on from Blender and reinstall it. If using the Blender Development add-on with Visual Studio Code, running the "Blender: Start" command will automatically install the Ranato add-on into Blender.


### ISSUE
```
CMake Error at /usr/share/cmake-3.28/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
    Could NOT find Python (missing: Interpreter Development.Module)
```

#### REASON 
Could be missing python3.11-dev or the headers for whatever Python version you’re using for PYAC

#### FIX
``` bash
sudo apt install python3.11-dev
```

### ISSUE     
```
    -- Could NOT find BLAS (missing: BLAS_LIBRARIES)
    -- Could NOT find BLAS (missing: BLAS_LIBRARIES)
    -- Could NOT find LAPACK (missing: LAPACK_LIBRARIES)
        Reason given by package: LAPACK could not be found because dependency BLAS could not be found.
```
#### REASON
Missing development packages:
- `libblas-dev`
- `liblapack-dev`

#### FIX 
```bash 
sudo apt install  libblas-dev \  liblapack-dev \
```
