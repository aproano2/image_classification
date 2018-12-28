# Image Classifier with Deep Learning

The goal of this project is to design an image classifier using deep learning. The code implemented here depends on the [PyTorch platform](https://pytorch.org/).

The `Image Classifier Project.ipynb` file provides Jupyter Notebook with the full implementation of the project. This file was broken down into three Python files to allow for training and classifying via the command line.

The `helper.py` includes all the functions used for image processing, model training, prediction and result display. Both `train.py` and `predict.py` are Python programs that run the respective process. To run them, execute the following commands in the command line.


```
python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
```

This command will generate a model and store it in a file named `checkpoint`.

```
python predict.py input checkpoint --category_names cat_to_name.json
```
