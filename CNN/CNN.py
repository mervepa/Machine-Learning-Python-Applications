import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# Dataset 1
#--------------------------------------------------------------------
(X1_train, y1_train), (X1_test, y1_test) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X1_train, X1_test = X1_train / 255.0, X1_test / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X1_train[i])
    plt.xlabel(class_names[y1_train[i][0]])
plt.show()


model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10))
model1.summary()


model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history1 = model1.fit(X1_train, y1_train, epochs=10, 
                    validation_data=(X1_test, y1_test))


plt.plot(history1.history['accuracy'], label='accuracy')
plt.plot(history1.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss1, test_acc1 = model1.evaluate(X1_test,  y1_test, verbose=2)

print(test_acc1)

#--------------------------------------------------------------------
# Dataset 2
#--------------------------------------------------------------------
(X2_train, y2_train), (X2_test, y2_test) = datasets.mnist.load_data(path="mnist.npz")
y2_train
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X2_train[i])
    plt.xlabel(y2_train[i])
plt.show()

X2_train = X2_train / 255.0
X2_test = X2_test / 255.0
X2_train = X2_train.reshape(X2_train.shape[0],28,28,1)
X2_test = X2_test.reshape(X2_test.shape[0],28,28,1)
y2_train = to_categorical(y2_train)
y2_test = to_categorical(y2_test)

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(10))
model2.summary()

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(X2_train, y2_train, validation_data=(X2_test, y2_test), epochs=3)

plt.plot(history2.history['accuracy'], label='accuracy')
plt.plot(history2.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss2, test_acc2 = model2.evaluate(X2_test,  y2_test, verbose=2)
print(test_acc2)


#--------------------------------------------------------------------
# Dataset 3
#--------------------------------------------------------------------
(X3_train, y3_train), (X3_test, y3_test) = datasets.fashion_mnist.load_data()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X3_train[i])
    plt.xlabel(y3_train[i])
plt.show()

X3_train = X3_train/255.0
X3_test = X3_test/255.0
y3_train = to_categorical(y3_train)
y3_test = to_categorical(y3_test)

X3_train = X3_train.reshape(X3_train.shape[0],28,28,1)
X3_test = X3_test.reshape(X3_test.shape[0],28,28,1)

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.Flatten())
model3.add(layers.Dense(10))
model3.summary()

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history3 = model3.fit(X3_train, y3_train, validation_data=(X3_test, y3_test), epochs=3)

plt.plot(history3.history['accuracy'], label='accuracy')
plt.plot(history3.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss3, test_acc3 = model3.evaluate(X3_test,  y3_test, verbose=2)
print(test_acc3)

