import tensorflow as tf
from tensorflow.keras import layers, models

# trains a convolutional neural network to read handwritten digits (MNIST).
# this is the "real" deep learning version, gets ~99% accuracy.
# pip install tensorflow


def build_model():
    # a small CNN: two conv layers, pooling, then dense layers
    model = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),                  # dropout so it doesnt overfit
        layers.Dense(10, activation="softmax"),  # 10 digits = 10 outputs
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    # load mnist, it comes built in with keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # scale pixels to 0-1 and add the channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    model = build_model()
    model.summary()

    model.fit(x_train, y_train, epochs=3, batch_size=128,
              validation_split=0.1)

    # check it on the test set it never trained on
    loss, acc = model.evaluate(x_test, y_test)
    print(f"\ntest accuracy: {acc:.4f}")

    # save the trained model so predict.py can use it
    model.save("digit_model.keras")
    print("saved -> digit_model.keras")


if __name__ == "__main__":
    main()
