conda create --name neugraspnet python=3.8 
conda activate neugraspnet
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt

download: - https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
pip install ~/Downloads/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl

sudo apt-get install libopenblas-dev libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.13.0
python setup.py install --user

pip install torch-scatter==2.1.0

cd ~/neugraspnet
pip install -e .
python neugraspnet/scripts/convonet_setup.py build_ext --inplace

pip install "urllib3<2"
pip install joblib
pip uninstall pybullet -y
pip install --no-cache-dir --force-reinstall --no-binary pybullet pybullet==2.7.9