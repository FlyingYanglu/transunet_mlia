# TransUNet Image Segmentation Project

This project aims to implement the TransUNet architecture for image segmentation. TransUNet is a transformer-based model that has shown promising results in various computer vision tasks, eapecially image segmentation. This is a final project of image analysis for machine learning at University of Virginia. We adapt tumor segmentation dataset to TransUNet. 

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In this project, we leverage the power of transformers to perform image segmentation using the TransUNet architecture. The TransUNet model combines the strengths of transformers and u-net to achieve state-of-the-art performance in image segmentation tasks.

## Installation

To get started with this project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/transunet-image-segmentation.git`
2. If running on rivanna, please following the below procedure:
    a. create new conda environment by `conda create --name [your env name] python=3.11`
    b. install pytorch with cuda 11.8 by `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
    c. install requirements by `pip install -r TransUNet/huiboreq.txt`
    d. certain problem with simplestk may need to echo the path of libgcc.
3. If running on specific machine, please follow their original [installation guides](https://github.com/Beckschen/TransUNet).


## Usage

To check the test set performance, please follow these steps. 

1. Prepare dataset. You should have the TumorSegmentation_data dataset ready. Exampler preparation outputs could be found inside `data\tumor\Testing_cases` folder. We provide two ways you could use to prepare the dataset.
    - Copy datset manually to target position
        1. Copy `TumorSegmentation_data/Segmentation_data/Testing/Brains` and `TumorSegmentation_data/Segmentation_data/Testing/Labels` folder to `data/tumor/raw/` folder. Please check the `data/tumor/raw/` for example data.
        2. cd into TransUNet
        3. run `python prepare_dataset.py`
    - Specify testing folder path 
        1. cd into TransUNet
        2. run `python prepare_dataset.py --data_path [path of TumorSegmentation_data/Segmentation_data/Testing ]`

    After preparing dataset, you should see Testing_cases have 19 cases if you are using tumorsegmentation dataset
2. Download model checkpoint [here]. Unzip the file and put the content of it into the model folder. After unzipping and data preparing, your folder should look like this.
    Transunet_mlia
    ├── data/tumor
    │   ├── lists
    │   ├── raw
    │   ├── Testing
    │   ├── Testing_cases
    ├── model
    │   ├── TU_tumor224
    │   ├── vit_checkpoint

    .....

3. run `python test.py --dataset tumor --vit_name R50-ViT-B_16`. You could add `--is_savenii` to save image results. Please use software to visualize them or check check_dataset.ipynb for demo of how to visualize results.


## Results

We have evaluated the TransUNet model with dice and hd95 metrics on Tumorsegmentation dataset. Please refer to this [report] for more info. 

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
