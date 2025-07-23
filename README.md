# Dog Breed Classifier 

---

## Features

* **Interactive Web UI:** Simple and intuitive interface powered by Streamlit.
* **Image Upload & Cropping:** Users can upload an image and use the `streamlit-cropper` to select the exact region of interest.
* **Top 5 Predictions:** Provides the top 5 most likely dog breeds instead of just one.
* **High Accuracy Model:** Built by training a state-of-the-art ResNet50V2 model.

---

## Technology Stack

* **Web Framework:** Streamlit
* **Deep Learning:** TensorFlow, Keras
* **Core Libraries:** NumPy, Pillow, streamlit-cropper

---

## Project Structure

```
Dog-Breed-Classifier/
│
├── Dog-Breed-Classifier-App/
│   └── Labels/
|       └── id_to_breed.json              # Maps model IDs to breed names.
|   └── Models/
│       ├── best_dog_breed_model.keras    # The best performing model saved during training.
│       ├── dog_breed_classifier.keras    # Latest saved model.
│   └── dog_breed_classifier.py       # The Streamlit application script.
│
├── Model Training/
│   └── dog_breed_classifier_model_training.py  # Script for training the model.
│
├── .gitattributes
├── LICENSE
├── README.md
└── requirements.txt                  # Python dependencies for the project.
```

---

## Getting Started

Follow these instructions to get the application running on your local machine.

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd Dog-Breed-Classifier-App
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Data Setup for Training

The model was trained on the **Stanford Dogs Dataset**, sourced from Kaggle. To run the training script (`dog_breed_classifier_model_training.py`), you must run on google colab or download and read the dataset.

---

## Usage

### 1. Running the Streamlit App

To start the web application, run the following command in your terminal. This will launch a new tab in your browser with the app.

```sh
streamlit run dog_breed_classifier.py
```

### 2. Training the Model

To re-train the model from scratch, ensure the dataset is set up correctly and run the training script:

```sh
python "Model Training/dog_breed_classifier_model_training.py"
```

---

## Model Information

* **Model Architecture:** The model is a **fine-tuned ResNet50V2** architecture, pre-trained on ImageNet, with a custom classification head for the 120 dog breeds.
* **Dataset:** Trained on the [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) from Kaggle.
* **Performance:** The model achieved the following final metrics on the test set:
    * **Test Accuracy:** **81.20%**
    * **Validation Accuracy:** **81.05%**.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
