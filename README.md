# Domain Generalized Gaze Estimation Network for Enhanced Cross-Environment Performance

The Pytorch Implementation of “Domain Generalized Gaze Estimation Network for  Enhanced Cross-Environment Performance”.

## Requirements

1. We bulid the project with python=3.8.

   ```python
   conda create -n GENet python=3.8
   ```

2. Clone the repo:

   ```python
   git clone https://github.com/yuhanghong123/GENet.git
   ```

3. Activate the environment:

   ```python
   conda activate GENet
   ```

4. Install the requirements:

   ```python
   cd GENet
   pip install -r requirements.txt
   ```

## Usage

We provides Res18-Version and Res50-Version. The Res50-Version is in the folder of  "utils/".

## Get Started

1. you can find data processing code from [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#eth-xgaze).

2. modifing files in  "config/" folder, modifing the model file in "Res18/" or "utils/" and run commands like:

   **Training and Test ETH—XGaze :** 

   ```shell
   bash total_eth.sh 
   ```

   **Training and Test Gaze360 :**

   ```shell
   bash total_gaze.sh
   ```

## Dataset

The datasets used in this study include [ETH-XGaze](https://ait.ethz.ch/xgaze) , [MPIIGaze](https://ieeexplore.ieee.org/abstract/document/8122058), [EyeDiap](https://www.idiap.ch/en/scientific-research/data/eyediap), and [Gaze360](https://gaze360.csail.mit.edu/).

If you use these datasets, please cite:

```
@inproceedings{Zhang_2020_ECCV,
    author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and Derek Bradley and Siyu Tang and Otmar Hilliges},
    title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation},
    year      = {2020},
    booktitle = {The European Conference on Computer Vision (ECCV)}
}
```



```
@article{zhang2017mpiigaze,
  title={Mpiigaze: Real-world dataset and deep appearance-based gaze estimation},
  author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={41},
  number={1},
  pages={162--175},
  year={2017},
  publisher={IEEE}
}
```



```
@inproceedings{eyediap,
    author = {Funes Mora, Kenneth Alberto and Monay, Florent and Odobez, Jean-Marc},
    title = {EYEDIAP: A Database for the Development and Evaluation of Gaze Estimation Algorithms from RGB and RGB-D Cameras},
    year = {2014},
    isbn = {9781450327510},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/2578153.2578190},
    doi = {10.1145/2578153.2578190},
    booktitle = {Proceedings of the Symposium on Eye Tracking Research and Applications},
    pages = {255–258},
    numpages = {4},
    keywords = {natural-light, database, RGB-D, RGB, remote sensing, gaze estimation, depth, head pose},
    location = {Safety Harbor, Florida},
    series = {ETRA '14}
}
```



```
@InProceedings{Kellnhofer_2019_ICCV,
    author = {Kellnhofer, Petr and Recasens, Adria and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
    title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Acknowledgement

- This code is adapted from [PureGaze](https://github.com/yihuacheng/PureGaze).
- We thank yihuacheng for elegant and efficient code base.
