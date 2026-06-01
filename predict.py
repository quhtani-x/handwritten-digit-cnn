import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# loads the trained model and guesses the digit in an image.
# the image should be a digit on a dark background, any size (it resizes).
# run train.py first to make digit_model.keras
# pip install tensorflow pillow numpy


def main():
    if len(sys.argv) < 2:
        print("usage: python predict.py my_digit.png")
        return

    model = tf.keras.models.load_model("digit_model.keras")

    # open the image, make it grayscale 28x28 like mnist
    img = Image.open(sys.argv[1]).convert("L").resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    # the model gives 10 probabilities, pick the biggest
    pred = model.predict(arr)
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))
    print(f"i think its a {digit} ({confidence*100:.1f}% sure)")


if __name__ == "__main__":
    main()
