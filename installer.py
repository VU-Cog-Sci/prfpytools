#wrapper
import os

assert "Python 3." in os.popen("python -V").read(), "Get python 3"

os.popen("conda config --add channels conda-forge").read()

print("Installing dependencies with conda...")
os.popen(f"conda install --yes numpy scipy cython h5py matplotlib ipython dill tqdm yaml pyyaml cifti nibabel nilearn joblib scikit-learn statsmodels cmasher").read()

assert "prfpy" in os.popen("conda list").read(), "Please install prfpy manually. (git clone https://github.com/VU-Cog-Sci/prfpy.git; cd prfpy; python installer.py)"
assert "pycortex" in os.popen("conda list").read(), "Please install pycortex manually. (git clone https://github.com/gallantlab/pycortex.git; cd pycortex; python -m pip install -e .)"

print("Installing prfpytools and verifying dependencies with pip...")
os.popen("python -m pip install -e .").read()
