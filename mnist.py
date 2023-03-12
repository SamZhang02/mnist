from matplotlib import pyplot as plt
from numpy import argmax, mean, std
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import time 

def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test= x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test

def prep_pixels (train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def show_data(x_train):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.show()

def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories

def run_test_harness():
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, x_test = prep_pixels(x_train, x_test)

    model = define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    model.save('model.h5')

    model = tf.keras.models.load_model('model.h5')
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))

def load_image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

def predict_digit(filename):
    img = load_image(filename)
    model = tf.keras.models.load_model('model.h5')
    digit = argmax(model.predict(img))
    print(digit)

def main():
    # TF config
    tf.config.set_visible_devices([], 'GPU')

    # Generate model and save in working directory
    t = time.time()
    run_test_harness()
    print('Time taken: ', time.time() - t)

    # Load model and evaluate a single image file in working directory
    predict_digit('test.png')

if __name__ == "__main__":
    main()
