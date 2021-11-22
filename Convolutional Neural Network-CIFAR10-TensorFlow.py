import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

model = models.Sequential([
    layers.Conv2D(filters=6, kernel_size=(5, 5),strides=(1, 1), padding='valid',activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid'),
    layers.Conv2D(filters=16, kernel_size=(5, 5),strides=(1, 1), padding='valid', activation='relu'),  #input_shape=(14, 14, 6)
    layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='valid'),
    layers.Flatten(),
    layers.Dense(units=120, activation='relu'),
    layers.Dense(units=84, activation='relu'),
    layers.Dense(units=10)])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
