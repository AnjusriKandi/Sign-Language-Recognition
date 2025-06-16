
#  Sign Language Recognition using CNN

<p align="justify">This project is a basic implementation of a Sign Language Recognition System using a Convolutional Neural Network (CNN). The goal is to recognize hand gestures representing alphabet signs (A–Z) in American Sign Language, helping bridge communication gaps between the hearing and speech-impaired communities and others.</p>


## 📌 Features

- 🧠 Built with a **Convolutional Neural Network** for image classification
- 📷 Takes **static hand gesture images** as input
- 🎯 Predicts and classifies gestures representing **English alphabets (A–Z)** except J and Z which require hand movement
- 📊 Trained and tested on a dataset of labeled hand sign images
- 🔍 Simple preprocessing pipeline with image resizing, grayscale conversion, and normalization


## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib / Seaborn** (for visualizations)


## 📁 Dataset

The project uses a publicly available dataset of sign language images [dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).

- The dataset contains images for each letter A–Z except J and Z as they require hand movement
- Preprocessed for uniform size and grayscale format


## 🚀 How It Works

1. **Data Loading** – Load and preprocess images (resizing, normalization).
2. **Model Building** – Create a CNN using Keras with Conv2D, MaxPooling, Flatten, and Dense layers.
3. **Training** – Train the model using the training set with categorical crossentropy loss.
4. **Evaluation** – Evaluate accuracy using validation/test data.
5. **Prediction** – Use the trained model to predict new sign images.


## 🧪 Sample Model Architecture

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(3, 3),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(filters=64,
                           kernel_size=(3, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(24, activation='softmax')
])
```

## 📊 Results
- Achieved 87% accuracy on the validation set.
- Confusion matrix and accuracy/loss plots included for performance analysis.

## 📌 Future Improvements
- Add real-time gesture recognition using webcam
- Extend support for dynamic gestures and words
- Incorporate Indian Sign Language grammar structure
- Optimize model using transfer learning

## 🤝 Contributing
Feel free to fork this repo, raise issues, or open pull requests to contribute!

