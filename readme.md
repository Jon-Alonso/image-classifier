# Deep Neural Network Image Classifier

This repository contains some of the files I used for my Artificial Intelligence Programming With Python Nanodegree at Udacity.

The project consisted in training a neural network on a image dataset and predicting the content of new images based on what it learned.
During the development stage the data was visualized using Matplotlib, Anaconda and Jupyter Notebooks.

### The main tools used were

| Library | Website |
| ------ | ------ |
| PyTorch | [http://pytorch.org][pytorch] |
| NumPy | [http://www.numpy.org][numpy] |
| Matplotlib | [https://matplotlib.org][matpl] |

### Usage
Install the dependencies all the dependencies and run:

#####  Training
Specify the path to the directory with the images.
```sh
$ python train.py data_directory_path
```

#####  Predicting
Specify the path to the image to be predicted and to the saved model.
```sh
$ python predict.py image_file_path checkpoint_path
```


####  Data visualization
Example code used during development and testing:
```python
def visualize(img_path, predictions, labels, label_mappings):
    true_label = img_path.split('/')[-2]
    predicted_range = np.arange(len(predictions))
    fig, (img_plot, label_plot) = plt.subplots(figsize=(12, 4), ncols=2, nrows=1)

    img_plot.set_xticks([])
    img_plot.set_yticks([])
    img_plot.set_title(label_mappings[true_label].capitalize())
    img_plot.imshow(Image.open(img_path))

    label_plot.set_yticks(predicted_range)
    label_plot.set_yticklabels([label_mappings[label] for label in labels])
    label_plot.invert_yaxis()
    label_plot.barh(predicted_range, predictions)
```
The following lines:
```python
predictions, labels = predict('flowers/test/78/image_01848.jpg', loaded_model)
visualize('flowers/test/78/image_01848.jpg', predictions, labels, cat_to_name)
```

Will output the following results on a properly trained model:

![Prediction](https://raw.githubusercontent.com/JohnAlonso/image-classifier/master/assets/prediction-sample.png)

[pytorch]: <http://pytorch.org][pytorch>
[numpy]: <http://www.numpy.org>
[matpl]: <https://matplotlib.org>