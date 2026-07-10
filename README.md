# Fake Product Review Detection Using Machine Learning

This is an academic mini-project to predict whether a product review is **Computer Generated (Fake)** or **Original (Genuine)** using a Machine Learning model (Logistic Regression).

## Folder Structure

* `dataset/` - Contains the dataset (`fake_review_dataset.csv`).
* `model/` - Contains the training script and the generated model files.
* `gui.py` - The main prediction application (GUI).
* `utils.py` - Text cleaning utilities used by both training and prediction.
* `requirements.txt` - Required Python libraries.
* `screenshots/` - Folder for project demonstration screenshots.

## Installation

Install the required dependencies using:

```
pip install -r requirements.txt
```

## How to Run

### Stage 1: Model Training
The project uses a two-stage architecture. Before running the GUI, you must train the model. This will process the dataset and generate `model.pkl` and `tfidf.pkl` in the `model/` directory.

Run the training script:
```
python model/train_model.py
```

### Stage 2: Prediction Application (GUI)
Once the model is trained, you can launch the prediction GUI. The GUI does not perform any training, it only predicts based on the trained model.

Run the GUI:
```
python gui.py
```

## Features
- **Traditional Machine Learning**: Uses TF-IDF for vectorization and Logistic Regression for classification.
- **Simple GUI**: Developed with `tkinter` for an easy-to-use desktop interface.
- **Modular Code**: Code is logically separated into training, utilities, and GUI components for readability and explanation.
