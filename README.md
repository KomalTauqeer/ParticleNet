# ParticleNet

Implementation of the jet charge tagger network using [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570) tensorflow implementation.

------
## Setting up enviroment (Only for once!)
Make sure to have miniconda3 installed and conda enviroment variables are set before you do the next steps.

For this setup, we specifically use python 3.6.8, to use the ROOT version 6.24.08 which is built for this python version and is available on etp machines.

1. Create a new environment with python
```conda create --name=tf_py36 -c conda-forge python==3.6.8```

2. Install tensorflow-gpu using pip
```pip install tensorflow-gpu==2.1.0```

3. Install CUDA and cuDNN compatible with tf
```conda install -c conda-forge cudatoolkit=10.1 cudnn=7.6.5```

4. Force-reinstallation of h5py to make tf, keras, python and h5py versions compatible
```pip install h5py==2.10.0 --force-reinstall```

5. Install other packages
```pip install uproot4 awkward pandas matplotlib scikit-learn table uproot-methods tqdm nvidia-cublas-cu11 nvidia-cudnn-cu11```

6. Enviroment variables for cuDNN
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

7. Include following lines in ~/.bashrc to use cuda.
```
cuda_init() {

    MYCUDAVERSION="cuda"
    export PATH="/usr/local/$MYCUDAVERSION/bin":$PATH
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib":${LD_LIBRARAY_PATH}
}

```
## How to use the model

#### MXNet model

The ParticleNet model can be obtained by calling the `get_particle_net` function in [particle_net.py](mxnet/particle_net.py), which can return either an MXNet `Symbol` or an MXNet Gluon `HybridBlock`. The model takes three input arrays:
 - `points`: the coordinates of the particles in the (eta, phi) space. It should be an array with a shape of (N, 2, P), where N is the batch size and P is the number of particles.
 - `features`: the features of the particles. It should be an array with a shape of (N, C, P), where N is the batch size, C is the number of features, and P is the number of particles.
 - `mask`: a mask array with a shape of (N, 1, P), taking a value of 0 for padded positions.

To have a simple implementation for batched training on GPUs, we use fixed-length input arrays for all the inputs, although in principle the  ParticleNet architecture can handle variable number of particles in each jet. Zero-padding is used for the `points` and `features` inputs such that they always have the same length, and a `mask` array is used to indicate if a position is occupied by a real particle or by a zero-padded value.

The implementation of a simplified model, ParticleNet-Lite, is also provided and can be accessed with the `get_particle_net_lite` function.

#### Keras/TensorFlow model

The use of the Keras/TensorFlow model is similar to the MXNet model. A full training example is available in [tf-keras/keras_train.ipynb](tf-keras/keras_train.ipynb).

## Citation
If you use ParticleNet in your research, please cite the paper:

	@article{Qu:2019gqs,
	      author         = "Qu, Huilin and Gouskos, Loukas",
	      title          = "{ParticleNet: Jet Tagging via Particle Clouds}",
	      year           = "2019",
	      eprint         = "1902.08570",
	      archivePrefix  = "arXiv",
	      primaryClass   = "hep-ph",
	      SLACcitation   = "%%CITATION = ARXIV:1902.08570;%%"
	}

## Acknowledgement
The ParticleNet model is developed based on the [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) model. The implementation of the EdgeConv operation in MXNet is adapted from the author's TensorFlow [implementation](https://github.com/WangYueFt/dgcnn), and also inspired by the MXNet [implementation](https://github.com/chinakook/PointCNN.MX) of PointCNN.
