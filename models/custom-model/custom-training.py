import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the path for the dataset and where the model will be saved
dataset_path = '/Users/dev_lap01/Downloads/garbage_classification'
model_save_path = '/Users/dev_lap01/Downloads/ecoDetect---thesis/models/custom-model/custom-model.keras'

# Image and training parameters
image_size = (224, 224)
batch_size = 32
epochs = 10

# Data preparation with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Custom CNN Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=image_size + (3,)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training process optimization
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Evaluate the model
model.load_weights(model_save_path)
evaluation = model.evaluate(validation_generator)
print(f'Loss: {evaluation[0]}, Accuracy: {evaluation[1]}')

# Prepare the data for confusion matrix
validation_generator_for_cm = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False 
)

# Reset the generator to ensure the order is the same as when labels were extracted
validation_generator_for_cm.reset()
Y_pred = model.predict(validation_generator_for_cm)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator_for_cm.classes

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
accuracy = np.trace(cm) / np.sum(cm)
print(f'Accuracy from confusion matrix: {accuracy * 100:.2f}%')

# Save the confusion matrix as a numpy file
np.save('/Users/dev_lap01/Downloads/ecoDetect---thesis/models/custom-model/confusion_matrix_custom.npy', cm)

# Class labels and plotting
class_labels = list(train_generator.class_indices.keys())
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Custom CNN - Confusion Matrix')
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Custom CNN - Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(1, epochs + 1))
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, epochs + 1), history.history['val_loss'], label='Validation Loss')
plt.title('Custom CNN - Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs + 1))
plt.legend()

plt.tight_layout()
plt.show()

# Get the classification report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save the classification report
report_df.to_csv('/Users/dev_lap01/Downloads/ecoDetect---thesis/models/custom-model/classification_report_custom.csv')

# Plotting the classification report with updated design
plt.figure(figsize=(10, 6))
# Transpose the DataFrame to switch the metrics to the top and labels to the left
report_df_t = report_df.iloc[:-1, :].transpose()
# Use a different colormap and format the annotations to avoid confusion with the confusion matrix
sns.heatmap(data=report_df_t, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.title('Custom CNN - Classification Report')
plt.xlabel('Metrics')  # Now on top
plt.ylabel('Classes')  # Now on the left
plt.show()
