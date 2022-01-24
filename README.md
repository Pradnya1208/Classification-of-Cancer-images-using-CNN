<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Classification of Cancer using CNN (IDC vs Non-IDC Images)</div>
<div align="center"><img src = "https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/intro.gif?raw=true" width="60%"></div>


## Overview:
Breast cancer is the most common form of cancer in women, and invasive ductal carcinoma (IDC) is the most common form of breast cancer. Accurately identifying and categorizing breast cancer subtypes is an important clinical task, and automated methods can be used to save time and reduce error.

## Dataset:
The dataset consists of 5547 breast histology images each of pixel size 50 x 50 x 3. The goal is to classify cancerous images (IDC : invasive ductal carcinoma) vs non-IDC images. In a first step we analyze the images and look at the distribution of the pixel intensities. Then, the images are normalized and we try out some basic classification algorithms like logistic regregession, random forest, decision tree and so on. We validate and compare each of these base models. After that we implement the following neural network architecture:
- input layer: [., 50, 50, 3]
- layer: Conv1 -> ReLu -> MaxPool: [., 25, 25, 36]
- layer: Conv2 -> ReLu -> MaxPool: [., 13, 13, 36]
- layer: Conv3 -> ReLu -> MaxPool: [., 7, 7, 36]
- layer: FC -> ReLu: [., 576]
- output layer: FC -> ReLu: [., 2]


## Implementation:

**Libraries:**  `NumPy`  `pandas` `sklearn`  `Matplotlib` `tensorflow` `keras`


## Data Exploration:
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/eda.PNG?raw=true">

### pixel intensity:
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/pixel%20intensity.PNG?raw=true">

### Augmented Data:
```
def generate_images(imgs):
    
    # rotations, translations, zoom
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 10, width_shift_range = 0.1 , height_shift_range = 0.1,
        zoom_range = 0.1)

    # get transformed images
    imgs = image_generator.flow(imgs.copy(), np.zeros(len(imgs)),
                                batch_size=len(imgs), shuffle = False).next()    
    return imgs[0]
```
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/augmented%20data.PNG?raw=true">

### Model training and Evaluation - ML models:
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/ml.PNG?raw=true">

### Neural Network:
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/nn.PNG?raw=true">
<img src="https://github.com/Pradnya1208/Classification-of-Cancer-images-using-CNN/blob/main/output/cm.PNG?raw=true" width="60%">




### Keras embedding layer:
A Keras Embedding Layer can be used to train an embedding for each word in your vocabulary. Each word (or sub-word in this case) will be associated with a 16-dimensional vector (or embedding) that will be trained by the model.
```
# Create an embedding layer.
embedding_dim = 16
embedding = tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim)
# Configure the embedding layer as part of a keras model.
model = tf.keras.Sequential(
    [
        embedding, # The embedding layer should be the first layer in a model.
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
```
### Saving the data for tensorboard:
TensorBoard reads tensors and metadata from the logs of your tensorflow projects. The path to the log directory is specified with log_dir below. For this tutorial, we will be using /logs/imdb-example/.

In order to load the data into Tensorboard, we need to save a training checkpoint to that directory, along with metadata that allows for visualization of a specific layer of interest in the model.
<br>
Checkout the [Notebook](https://github.com/Pradnya1208/Visualizing-data-using-the-embedding-projector-in-Tensorboard/blob/main/docs/tensorboard_projector_plugin.ipynb) for implementation details.

### Run tensorboard against on log data we just saved:
%tensorboard --logdir /logs/imdb-example/

The TensorBoard Projector is a great tool for interpreting and visualzing embedding. The dashboard allows users to search for specific terms, and highlights words that are adjacent to each other in the embedding (low-dimensional) space.



### Learnings:
`Word embeddings` `embedding layers` `tensorboard` 






## References:
[Predicting IDC in Breast Cancer Histology Images](https://www.kaggle.com/paultimothymooney/predicting-idc-in-breast-cancer-histology-images/notebook)

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]
