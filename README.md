# Cell segmentation using U-Net based networks

## Project Pipeline   

> data > preprocessing > models > training > predictions

<img src="https://user-images.githubusercontent.com/76240694/192287299-3f67b4fd-c844-4398-aa33-6a4717ffd59d.png" width="600">

## GPU Support   

Guidance for Tensorflow Installation with CUDA, cudNN and GPU support: [Youtube Video](https://www.youtube.com/watch?v=hHWkvEcDBO0)

> Tested on Windows environment with Tensorflow 2.9, CUDA 11.2, cudnn 8.1   
> Only applicable for Tensorflow <= 2.10   
> For recent Tensorflow, please refer [link](https://www.tensorflow.org/install/pip?_gl=1*1jwqv1w*_ga*ODI3Mjk2MjIwLjE3MDIyNjkyNTI.*_ga_W0YLR4190T*MTcwMjU5Njk4OS43LjEuMTcwMjU5NzMyNS4wLjAuMA..#windows-wsl2_1)

![image](https://github.com/Fyzie/Cell-segmentation-using-U-Net-based-networks/assets/76240694/c79964af-caef-425b-9170-dc7f24ed8186)


1. [GO HERE FIRST](https://www.tensorflow.org/install/source#gpu) to check cross compatibility

2. Download and install [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/community/)

3. Download and install [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

4. Download [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

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

### Possible Errors
```
Could not locate zlibwapi.dll. Please make sure it is in your library path
```
Copy of the missing zlib DLL in the NVIDIA Nsight directory:

> C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.4.2\host-windows-x64\zlib.dll   

Copied and renamed it to:

> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\zlibwapi.dll   
