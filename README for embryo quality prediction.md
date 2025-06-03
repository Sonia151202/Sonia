# IVF Embryo Classification using DenseNet121 Transfer Learning

This project performs multi-class classification of IVF embryo images using a fine-tuned DenseNet121 convolutional neural network. The model classifies embryo images into 10 classes using TensorFlow and Keras.

---

## Features

- Programmatic Kaggle dataset import using `kagglehub`
- Dataset splitting into train, validation, and test sets using `splitfolders`
- Image preprocessing and data augmentation with `ImageDataGenerator`
- Transfer learning with pretrained DenseNet121 (ImageNet weights)
- Two-stage training:
  - Stage 1: Train only the top classifier layers
  - Stage 2: Fine-tune the full model with a low learning rate
- Evaluation using accuracy, classification report, and confusion matrix visualization
- Model saving for future inference

---

## Setup & Installation

Install required Python packages:
```bash
pip install tensorflow seaborn matplotlib split-folders kagglehub
```

---

## Dataset

- Dataset downloaded from Kaggle: `bommasonia/final-ivf-train`
- Directory structure assumed:
```
final_ivf_train/
├── class_0/
├── class_1/
├── ...
└── class_9/
```
- The dataset is split into train (70%), validation (10%), and test (20%) sets using `splitfolders`.

---

## Usage

### 1. Import Kaggle dataset

```python
import kagglehub
kagglehub.login()
bommasonia_final_ivf_train_path = kagglehub.dataset_download('bommasonia/final-ivf-train')
```

### 2. Split dataset into train/val/test folders

```python
import splitfolders

input_folder = "/kaggle/input/final-ivf-train/final_ivf_train"
output_folder = "/kaggle/working/train_test_val_split"

splitfolders.ratio(input_folder, output=output_folder,
                   seed=42, ratio=(0.7, 0.1, 0.2),
                   move=False)
```

### 3. Create data generators

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    output_folder + '/train', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=True)

validation_generator = val_datagen.flow_from_directory(
    output_folder + '/val', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=False)

test_generator = test_datagen.flow_from_directory(
    output_folder + '/test', target_size=(224, 224),
    batch_size=32, class_mode='categorical', shuffle=False)
```

### 4. Build and compile the model

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
import tensorflow as tf

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 5. Train top layers

```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

### 6. Fine-tune entire model

```python
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

### 7. Evaluate on test set

```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

### 8. Save model

```python
model.save('densenet121_fine_tuned_model.h5')
```

### 9. Predict and analyze results

```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)
predicted_class_indices = np.argmax(predictions, axis=1)
true_class_indices = test_generator.classes

# Trim to equal length if necessary
min_len = min(len(predicted_class_indices), len(true_class_indices))
predicted_class_indices = predicted_class_indices[:min_len]
true_class_indices = true_class_indices[:min_len]

class_labels = list(test_generator.class_indices.keys())

print("Classification Report:")
print(classification_report(true_class_indices, predicted_class_indices, target_names=class_labels))

conf_matrix = confusion_matrix(true_class_indices, predicted_class_indices)
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
```

---

## Author

**Bomma Sonia**  
Data Scientist | AI & Deep Learning Enthusiast

---

## License

This project is licensed under the MIT License.

