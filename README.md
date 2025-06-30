<<<<<<< HEAD
# FABRIC-PATTERN-CLASSIFICATION-WITH-DEEP-LEARNING
=======
# Fabric Pattern Recognition

This project is a web application that uses a PyTorch deep learning model to classify fabric patterns from an uploaded image.

## Quick Start

1.  **Train the model (run this only once):**
    ```bash
    python helpers.py
    ```

2.  **Run the application:**
    - **Web App:**
      ```bash
      python app.py
      ```
      Then open your browser to `http://127.0.0.1:5000`.
    - **Command-Line Prediction:**
      ```bash
      python predict.py "path/to/your/image.jpg"
      ```

## How to Use

There are three main steps: training the model, running the web application, and predicting via command line.

### Step 1: Train the Model (Only needs to be done once)

Before you can run the web application, you must train the model. This process analyzes the images in the `pattern-recognition/train` directory and creates a `fabric_pattern_model.pth` file.

Open your terminal and run the following command:
```bash
python helpers.py
```
You only need to do this once. You would only re-run this step if you add more images to your dataset or change the model architecture in `helpers.py`.

### Step 2: Run the Web Application

Once the model is trained and `fabric_pattern_model.pth` exists, you can start the web application.

Run this command in your terminal:
```bash
python app.py
```
Or, on Windows, you can simply double-click the `run_app.bat` file.

Open your web browser and navigate to `http://127.0.0.1:5000` to use the application.

### Step 3: Predict via Command Line (Alternative)

As an alternative to the web application, you can classify an image directly from your terminal.

Run the `predict.py` script followed by the path to your image:
```bash
python predict.py "path/to/your/image.jpg"
```

## File Descriptions

- `helpers.py`: The script to train the neural network and save the model.
- `app.py`: The Flask web server that loads the saved model and handles predictions.
- `predict.py`: A command-line tool to classify a single image.
- `check_dataset.py`: A utility script to check the image dataset.
- `templates/index.html`: The HTML frontend for the web application.
- `fabric_pattern_model.pth`: The saved, trained model file (created by `helpers.py`).
- `run_app.bat`: A helper script for Windows to easily start the web app.
>>>>>>> 99ece12 (Initial commit for Fabric Pattern Classifier)
