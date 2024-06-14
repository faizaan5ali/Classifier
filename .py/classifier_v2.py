import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Data directories
BASE_DIR = 'data/'
subfolders = next(os.walk(BASE_DIR + 'train/'))[1]
print(subfolders)

# Data augmentation and generators
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.5,
    brightness_range=[0.5, 1.0],
)
valid_gen = ImageDataGenerator(rescale=1. / 255)
test_gen = ImageDataGenerator(rescale=1. / 255)

target_size = (64, 64)

train_batches = train_gen.flow_from_directory(
    'data/train',
    target_size=target_size,
    class_mode='categorical',
    batch_size=128,
    shuffle=True,
    color_mode="grayscale",
    classes=subfolders
)

valid_batches = valid_gen.flow_from_directory(
    'data/val',
    target_size=target_size,
    class_mode='categorical',
    batch_size=128,
    shuffle=False,
    color_mode="grayscale",
    classes=subfolders
)

test_batches = test_gen.flow_from_directory(
    'data/test',
    target_size=target_size,
    class_mode='categorical',
    batch_size=128,
    shuffle=False,
    color_mode="grayscale",
    classes=subfolders
)


model = Sequential()
model.add(Conv2D(512, 3, input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(512, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(200, activation='softmax'))


# Compile the model
sgd_optimizer = SGD(learning_rate=0.005, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd_optimizer, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(train_batches, validation_data=valid_batches, epochs=50, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_batches)
print('Test accuracy:', test_acc)

# Save the model
model.save("lego_model_4layer_bw_v3.h5")

# Plot loss and accuracy
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15)
plt.show()


# Preprocessing function for new images
def preprocess_image(image_p):
    img = tf.keras.preprocessing.image.load_img(image_p, target_size=(64, 64), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array


# Define the target size (assuming your model expects images of size 64x64)
target_size = (64, 64)

# Load and preprocess the image
image_path = 'data/test_6091.jpeg'
image = load_img(image_path, target_size=target_size, color_mode="grayscale")
# image.save('data/test_3002 (2).jpeg')

# image = Image.open('data/test_3002.jpeg').convert('L')
image_array = img_to_array(image)
image_array = image_array / 255.0  # Normalize pixel values
plt.imshow(image_array)
# Add an extra dimension to the image array to match the input shape expected by the model
image_array = np.expand_dims(image_array, axis=0)
print(image_array.shape)

# Perform inference
predictions = model.predict(image_array)

# Get top 3 predicted classes and their probabilities
top_classes_idx = np.argsort(predictions[0])[-3:][::-1]  # Indices of top 3 classes
top_classes_prob = predictions[0][top_classes_idx]  # Probabilities of top 3 classes
top_classes_labels = [subfolders[i] for i in top_classes_idx]  # Labels of top 3 classes

# Show the top 3 predicted classes and their probabilities
for label, prob in zip(top_classes_labels, top_classes_prob):
    print(f"Class: {label}, Probability: {prob:.4f}")