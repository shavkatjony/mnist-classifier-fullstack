# MNIST Neural Network Web Application

An end-to-end machine learning project that trains a neural network on the MNIST handwritten digit dataset and serves predictions through a full-stack web interface.

## Overview

This project demonstrates the complete workflow of a simple deep learning system:

- data loading and preprocessing
- neural network training
- model saving and inference
- backend API integration
- frontend user interface for digit drawing and prediction

## Features

- Train a neural network on the MNIST dataset
- Save and load the trained model
- Predict handwritten digits from a web interface
- Backend built with Python
- Frontend built with HTML, CSS, and JavaScript
- Clean project structure for experimentation and extension

## Project Structure

```text
mnist-neural-network/
├── backend/
│   ├── main.py
│   ├── model.py
│   └── requirements.txt
├── data/
├── static/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── models/
│   └── mnist_model.pth
├── run.sh
├── .gitignore
└── README.md
