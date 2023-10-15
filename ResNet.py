import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Here first we define the image dimensions and batch size
image_size = (224, 224)
batch_size = 32

# Then we create  a data generator for loading and we augment the data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'path_to_train_data_directory',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# We create a ResNet model with pre-trained weights (not including the top layers)
base_model = ResNet50(weights='imagenet', include_top=False)

# Then we add custom top layers for cat and dog classification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')  # O/P class : Cat and dog 
])

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# We Fine-tune the model on our dataset
model.fit(train_generator, epochs=10)  

#for any future use we will save the model here
model.save('cat_dog_resnet_model.h5')
