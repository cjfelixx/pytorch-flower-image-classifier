# pytorch-flower-image-classifier

Flower Image Classifier using Pytorch

This was a project from Udacity *AI Programming with Python Nanodegree*

## Setup

Create conda env

```bash
conda env create -f environment.yml
```

To edit files in Jupyter

```bash
jupyter lab
```

### Flower data

Can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

## Model training

```bash
python3 train.py <data directory>
```

## Model predicting

```bash
python3 predict.py <image_file_path>
```
