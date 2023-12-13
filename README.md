# Cell segmentation using U-Net based networks

Guidance for Tensorflow Installation with CUDA, cudNN and GPU support: [Youtube Video](https://www.youtube.com/watch?v=hHWkvEcDBO0)

> Windows environment   

1. [GO HERE FIRST](https://www.tensorflow.org/install/source#gpu) to check cross compatibility

2. Download and install [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/community/)

3. Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

4. Download and install [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

6. Extract cuDNN and transfer the files within
   - Copy folders bin, include, lib
   - Paste and replace to ...\NVIDIA GPU Computing Toolkit\CUDA\v11.2\
   - Search and open Edit the system environment variables
   - Go to Environment Variables
   - Double click on Path and add these full dir path:
     - ...\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
     - ...\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp

8. Install and create [Anaconda Python](https://www.anaconda.com/download) environment (check compatible version)
```
conda create --name {any_name} python=={compatible_version}
```
6. Install Tensorflow (compatible with GPU support)
```
pip install tensorflow=={compatible_version}
```

## Project Pipeline

<img src="https://user-images.githubusercontent.com/76240694/192287299-3f67b4fd-c844-4398-aa33-6a4717ffd59d.png" width="600">
