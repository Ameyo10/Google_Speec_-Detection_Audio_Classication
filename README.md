# Speech Detection Audio Classifier

This repository contains a Python notebook for building a speech command classification model using the [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). The model is implemented in PyTorch and classifies short audio clips into one of 35 possible spoken keywords.

## Features

- Downloads and preprocesses the Speech Commands dataset
- Extracts MFCC features from audio files
- Splits data into training, validation, and test sets
- Builds a deep convolutional neural network for audio classification
- Trains and evaluates the model, with loss/accuracy plots and confusion matrix
- Predicts classes for new audio samples

## Dataset

- **Source:** [Google Speech Commands Dataset v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands)
- **Description:** 105,000+ short utterances of 35 different words for keyword spotting tasks.

## Requirements

- Python 3.8+
- PyTorch
- librosa
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm
- torchmetrics
- mlxtend
- IPython

Install dependencies with:

```sh
pip install torch librosa pandas numpy scikit-learn matplotlib tqdm torchmetrics mlxtend ipython
```

## Usage

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/speech-detection-audio-classifier.git
    cd speech-detection-audio-classifier
    ```

2. **Run the notebook:**
    - Open [Speech_detection_audio_classifier.ipynb](f:/Python%20Projects/Learning/Neural%20Networks/Speech_detection_audio_classifier.ipynb) in Jupyter Notebook or VS Code.
    - Execute the cells in order. The notebook will:
        - Download and extract the dataset (if not already present)
        - Preprocess audio and extract MFCC features
        - Train and evaluate the model
        - Save the trained model to the `models/` directory

3. **Results:**
    - Training/validation loss and accuracy curves
    - Confusion matrix for validation set
    - Predictions for test audio samples

## Directory Structure

```
data/
    Speech_detection_dataset/
models/
    Speech_Detecting_Audio_Classifier.pth
Speech_detection_audio_classifier.ipynb
```

## Notes

- The notebook uses a subset of 10,000 random samples for training for faster experimentation.
- Model and dataset paths are configurable in the notebook.
- GPU acceleration is used if available.

## References

- [Google Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [librosa Documentation](https://librosa.org/doc/latest/index.html)

---

**Author:** Ameyo Jha  
**License:** MIT
