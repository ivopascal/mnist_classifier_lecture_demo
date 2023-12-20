# Demo for Machine Learning Practical lecture
This repository follows a lecture for the Machine Learning Practical.
It may be used as reference material on FastAPI and Streamlit.

For the actual project please follow the [original repository](https://github.com/arun477/mnist_classifier)


---
title: Mnist
emoji: 🐢
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
---

# <img src="/static/favicon.png" alt="Logo" style="float: left; margin-right: 10px; border-radius:100%;margin-top:5px" />  MNIST CLASSIFIER
MNIST classifier from scratch
* Model: CNN
* Accuracy: 97%

* Training Notebook: mnist_classifier.ipynb
  
* Cleaned Python Inference Version: mnist_classifier.py (This file is auto-generated from mnist_classifier.ipynb. Please do not edit it.)

* Model: classifier.pth

* Try Online: https://carlfeynman-mnist.hf.space/

* If you want to host this on Hugging Face as a space, please refer to [this documentation.](https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo)

* To Run FastApi Server Locally: uvicorn server:app --reload
  
* Note: When you try it out on the website, accuracy may drop due to distribution changes from training data to canvas image input. It has not been adjusted or fine-tuned for this specific purpose; it's intended just to demonstrate the full flow.
   
![site_screenshot](/static/site_screenshot.png)
