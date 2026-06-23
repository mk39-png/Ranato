# Ranato 

Blender add-on showcasing a practical application of PYAC for technical artists.


## Installation

Be sure to replace [Repos] with the location of where you store your repositories.

### Installs PYAC Python library with its dependencies (i.e. Cholespy external library)
git clone --recursive https://github.com/mk39-png/py-algebraic-contours.git

### OPTIONAL: install requirements into virtual environment of Ranato for typing and context
pip install -e [directory to py-algebraic-contours]
pip install -r requirements.txt [within the workspace folder of the Ranato addon]

### Install PYAC wheel into Ranato
pip wheel --no-cache-dir D:\[Repos]\py-algebraic-contours\ -w D:\[Repos]\Ranato\ranato\wheels\ 

### Install Cholespy wheel (to overwrite the Cholespy install from PyPi)
pip wheel --no-cache-dir D:\[Repos]\py-algebraic-contours\external\cholespy\ -w D:\[Repos]\Ranato\ranato\wheels\


## Running Ranato

For development, Ranato can be ran in Blender using the Blender Development add-on in Visual Studio Code.
https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development
