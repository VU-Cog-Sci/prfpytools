#wrapper
import os

assert "Python 3." in os.popen("python -V").read(), "Get python 3 you fool"

os.popen("conda config --add channels conda-forge").read()

print("Installing dependencies with conda...")
os.popen(f"conda install --yes numpy scipy h5py matplotlib tqdm yaml pyyaml cifti nibabel nilearn joblib scikit-learn statsmodels cmasher").read()

print("Installing prfpy_tools and verifying dependencies with pip...")
os.popen("python -m pip install -e .").read()
