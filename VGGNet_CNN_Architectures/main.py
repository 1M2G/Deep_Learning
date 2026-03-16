# ===============================
# SUPPRESS WARNINGS
# ===============================

import os                                  # Import OS module for file system operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TensorFlow warnings and info logs
import warnings                            # Import warnings module
warnings.filterwarnings("ignore")          # Disable Python warning messages

# ===============================
# IMPORT LIBRARIES
# ===============================

from tensorflow import keras                       # Import Keras API from TensorFlow
from tensorflow.keras import layers, models        # Import neural network layers and model tools
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import data augmentation generator
from tensorflow.keras.regularizers import l2       # Import L2 regularization
from sklearn.metrics import classification_report, confusion_matrix # Import evaluation metrics
import numpy as np                                 # Import NumPy for numerical operations
import matplotlib.pyplot as plt                    # Import Matplotlib for plotting
from PIL import Image                              # Import PIL to validate images

# ===============================
# CREATE FOLDER FOR VISUALS
# ===============================

os.makedirs('visuals', exist_ok=True)  # Create folder to store plots if it does not already exist


# ===============================
# FUNCTION TO REMOVE CORRUPTED IMAGES
# ===============================

def remove_corrupted_images(folder_path):  # Function that scans dataset folders and deletes bad images
    
    for root, dirs, files in os.walk(folder_path):  # Loop through all subfolders and files
        
        for file in files:  # Loop through each file
            
            file_path = os.path.join(root, file)  # Construct full file path
            
            try:
                img = Image.open(file_path)       # Try to open image file
                img.verify()                      # Verify that it is a valid image
                
            except:
                print("Removing corrupted file:", file_path)  # Print corrupted file name
                os.remove(file_path)                           # Delete corrupted file


# Remove corrupted images from both training and test datasets
remove_corrupted_images('Birds dataset/train')
remove_corrupted_images('Birds dataset/test')


# ===============================
# IMAGE SETTINGS
# ===============================

IMG_SIZE = (64, 64)     # Resize all images to 64x64 pixels
BATCH_SIZE = 32         # Number of images processed per batch
EPOCHS = 20             # Number of training epochs


# ===============================
# DATA AUGMENTATION
# ===============================

train_datagen = ImageDataGenerator(

    rescale=1./255,          # Normalize pixel values to range 0-1
    rotation_range=10,       # Rotate images randomly within 10 degrees
    width_shift_range=0.1,   # Shift images horizontally by up to 10%
    height_shift_range=0.1,  # Shift images vertically by up to 10%
    horizontal_flip=True     # Flip images horizontally

)


# ===============================
# LOAD TRAINING DATASET
# ===============================

train_data = train_datagen.flow_from_directory(

    'Birds dataset/train',   # Directory containing training images
    target_size=IMG_SIZE,    # Resize images to 64x64
    batch_size=BATCH_SIZE,   # Load images in batches
    class_mode='sparse'      # Labels are integers instead of one-hot vectors

)


# ===============================
# TEST DATA GENERATOR
# ===============================

test_datagen = ImageDataGenerator(

    rescale=1./255           # Only normalize images (no augmentation)

)


# ===============================
# LOAD TEST DATASET
# ===============================

test_data = test_datagen.flow_from_directory(

    'Birds dataset/test',    # Test dataset folder
    target_size=IMG_SIZE,    # Resize images
    batch_size=BATCH_SIZE,   # Batch size
    class_mode='sparse',     # Integer labels
    shuffle=False            # Do not shuffle test data (needed for evaluation)

)


# ===============================
# CLASS INFORMATION
# ===============================

class_names = list(test_data.class_indices.keys())  # Extract class names from dataset
num_classes = len(class_names)                      # Count number of classes

print("Classes:", class_names)                      # Display detected classes


# ===============================
# BUILD CUSTOM CNN MODEL
# ===============================

def build_custom_cnn():  # Function that builds a simple convolutional neural network

    model = keras.Sequential([

        layers.Input(shape=(*IMG_SIZE, 3)),  # Define input layer

        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=l2(0.001)),  # First convolution layer
        layers.MaxPooling2D(),                                                   # Downsample feature map
        layers.Dropout(0.25),                                                    # Drop neurons to prevent overfitting

        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=l2(0.001)),   # Second convolution layer
        layers.MaxPooling2D(),                                                   # Downsample again
        layers.Dropout(0.25),                                                    # Dropout

        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=l2(0.001)),  # Third convolution layer
        layers.MaxPooling2D(),                                                   # Downsample
        layers.Dropout(0.25),                                                    # Dropout

        layers.Flatten(),                                                        # Convert feature maps to vector

        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),     # Fully connected layer
        layers.Dropout(0.5),                                                     # Strong dropout

        layers.Dense(num_classes, activation='softmax')                          # Output layer

    ])

    model.compile(

        optimizer='adam',                       # Optimization algorithm
        loss='sparse_categorical_crossentropy', # Loss for multi-class classification
        metrics=['accuracy']                    # Track classification accuracy

    )

    return model                                # Return constructed model


# ===============================
# BUILD VGG-LIKE NETWORK
# ===============================

def build_vggnet():  # Function that builds simplified VGG-style architecture

    model = keras.Sequential([

        layers.Input(shape=(*IMG_SIZE, 3)),  # Input layer

        layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),

        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')

    ])

    model.compile(

        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']

    )

    return model


# ===============================
# EVALUATION FUNCTION
# ===============================

def eval_and_plot(model, history, name):  # Function that evaluates model and generates plots

    loss, acc = model.evaluate(test_data, verbose=0)  # Evaluate model on test dataset
    print(f"{name}: Loss={loss:.4f}, Accuracy={acc:.4f}")  # Print performance

    preds = model.predict(test_data, verbose=0)       # Generate predictions
    pred_classes = np.argmax(preds, axis=1)           # Convert probabilities to class labels
    true_classes = test_data.classes                  # True labels

    print(classification_report(true_classes, pred_classes, target_names=class_names))  # Print precision, recall, F1


    # ===============================
    # TRAINING HISTORY PLOTS
    # ===============================

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(name + " Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(name + " Loss")
    plt.legend()

    plt.savefig(f'visuals/{name}_history.png')
    plt.close()


    # ===============================
    # CONFUSION MATRIX
    # ===============================

    cm = confusion_matrix(true_classes, pred_classes)

    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap='Blues')
    plt.title(name + " Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(f'visuals/{name}_confusion.png')
    plt.close()


# ===============================
# TRAIN CUSTOM CNN
# ===============================

print("CUSTOM CNN")

custom = build_custom_cnn()        # Build model
print(custom.summary())            # Display architecture

history_custom = custom.fit(       # Train model
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

eval_and_plot(custom, history_custom, "Custom_CNN")  # Evaluate model


# ===============================
# TRAIN VGG NETWORK
# ===============================

print("\nVGGNet")

vgg = build_vggnet()               # Build VGG model
print(vgg.summary())               # Show architecture

history_vgg = vgg.fit(             # Train VGG model
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

eval_and_plot(vgg, history_vgg, "VGGNet")  # Evaluate VGG


print("All done. Results saved in visuals folder.")