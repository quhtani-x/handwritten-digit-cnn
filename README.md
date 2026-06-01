# Handwritten Digit CNN

A convolutional neural network that reads handwritten digits (MNIST), built
with TensorFlow/Keras. This is the proper deep-learning version — two conv
layers + pooling + dropout — and it hits about 99% accuracy.

`train.py` trains and saves the model, `predict.py` loads it and guesses the
digit in any image you give it.

## run

```bash
pip install tensorflow pillow numpy

python train.py            # trains and saves digit_model.keras
python predict.py mine.png # guess a digit from an image
```

## the model

```
Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> Flatten -> Dense(64) -> Dropout -> Dense(10, softmax)
```

tags: python, ai, ml, deep-learning, tensorflow, cnn

stepped up from the sklearn digits version to a real CNN, big accuracy jump.
