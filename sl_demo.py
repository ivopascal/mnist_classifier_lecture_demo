# Streamlit demo written by Ivo

import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import torch

import mnist_classifier


def process_image(file):
    image = Image.open(file)
    image = image.resize((28, 28))  # Resize to MNIST image size
    image = image.convert("L")  # Convert to grayscale
    raw_image = image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    return torch.from_numpy(image).float().reshape(1, 28, 28), raw_image


def main():
    st.title("MNIST image classifier")
    image = st.file_uploader(label="Image to classify")
    if not image:
        return
    tensor_image, raw_image = process_image(image)
    confs: np.ndarray = mnist_classifier.predict(tensor_image)
    st.write("# Predicted class confidences:")
    predictions = pd.DataFrame(data={"Confidence": confs})
    st.table(predictions)
    st.write("# Input image")
    st.image(image)


if __name__ == "__main__":
    main()
