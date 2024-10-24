# AdaptoML_UX
An Adaptive Incremental End-to-End ML Framework with Automated Feature Engineering

## Overview
AdaptoML_UX is an innovative machine learning framework that offers an intuitive graphical user interface. It's designed to streamline and automate key stages in the machine learning workflow, such as feature extraction, selection, and model adaptation, making it more accessible for users with varying levels of expertise in machine learning.

## Installation Instructions
To get started with AdaptoML_UX, ensure you have Python 3.x installed on your system. Follow these steps:
1. Clone the repository or download the source code.
2. Navigate to the project directory and set up a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Unix or MacOS: `source venv/bin/activate`
4. Install the required dependencies: `pip install -r requirements.txt`

## Code Structure
- `MainGUI.py`: Core file for the application's graphical interface.
- `ComputeOutput.py`: Manages the computation and processing of models.
- `Classifiers_Regressors.py`: Implements various machine learning models.
- Other files like `FeatureExtraction.py`, `FeatureSelection.py`, and `Imputation.py` handle respective functionalities in the ML pipeline.
- `Utilities.py`: Provides supporting functions used throughout the application.

## Usage Guide
Run `MainGUI.py` to start the application. The GUI is self-explanatory and user-friendly, guiding you through various steps of machine learning model development:

### Mandatory Arguments Section
<img src="logo_image/MA.png" width="800">

- **Select CSV data file**: Click the 'Browse' button to upload your dataset in CSV format.
- **Label column name**: Enter the name of the column that contains the target or label for your model.
- **Adaptation column name**: If your data includes a column for adaptation (e.g., user ID), specify it here.
- **This data is for (task type)**: Choose the type of machine learning task from the dropdown menu (classification, regression).
- **Reset inputs**: Click this button to clear all inputs and reset to default settings.

### Data Preprocessing Section
<img src="logo_image/DP.png" width="800">

- **Imputation method**: Choose how you want to handle missing data from the dropdown menu (simple, multivariate, KNNImputer).
- **Normalization type**: Select the normalization technique for your data (Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler).
- **Test size**: Specify what proportion of your data should be used for testing. Test size must be smaller than 1 and greater than 0, default 0.3.
- **Validation size**: Define the size of the validation set. Validation size must be smaller than 1 and greater than 0, default 0.3.
- **Columns to be initially removed**: List any columns you want to exclude from the model. You should leave space between each name.

### Extraction and Selection Section
<img src="logo_image/ES.png" width="800">

- **Feature extraction methods**: Include extraction methods you want to apply to the data set.
- **Feature selection methods**: selection extraction methods you want to apply to the data set
- **Perform feature selection first**: Toggle whether to perform feature selection before extraction.
- **Save processed feature as**: Decide if you want to save processed features as a `.csv` or `.pkl` file.
- **Metric to select best extraction/selection method with**: Select the metric for evaluating feature extraction and selection from the dropdown (accuracy, weighted_precision, weighted_recall, weighted_f1score", "r2_score, median_absolute_error, mean_squared_error, mean_absolute_error).

### Classifiers and Regressors Section
<img src="logo_image/clf.png" width="800">

- **Classifiers and regressors**: Enable or disable this module using the checkbox.
- **Select a suitable metric**: From the dropdown, choose the metric for adaptation, incremental learning, or evaluation (accuracy, weighted_precision, weighted_recall, weighted_f1score", "r2_score, median_absolute_error, mean_squared_error, mean_absolute_error).
- **Load trained models file/s**: Use the 'Browse' button to upload pre-trained model files.
- **Perform end-to-end model adaptation**: Check this option to adapt your model throughout the entire process.
- **Load previous adapted data (.pkl)**: Upload previously adapted data if available.
- **Train**: Tick this checkbox to train a new model.
- **Batch Incremental Learning (i.e., Partial Fit)**: Select data for batch incremental learning.
- **Predict**: Enable the prediction feature.
- **Evaluation**: Provides information on the model evaluation process.
- **Save models**: Decide whether to save the trained models using the dropdown. The models can be saved separately or together in one ine file.

These features are part of the AdaptoML-UX interface, designed to facilitate an interactive and user-friendly experience for machine learning model development.

## Issues
If you face any problems while using the toolkit, please open an issue here - https://github.com/MichaelSargious/AdaptoML_UX/issues or contact us under amr.gomaa@dfki.de.

## Contact Information
For support, queries, or contributions, please reach out to amr.gomaa@dfki.de.

## Citation

- If you find this code helpful, please cite our papers:
```

@misc{gomaa2024adaptomluxadaptiveusercenteredguibased,
      title={AdaptoML-UX: An Adaptive User-centered GUI-based AutoML Toolkit for Non-AI Experts and HCI Researchers}, 
      author={Amr Gomaa and Michael Sargious and Antonio Krüger},
      year={2024},
      eprint={2410.17469},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2410.17469}, 
}

```
