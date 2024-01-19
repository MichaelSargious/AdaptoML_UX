# AdaptoML_UX
An Adaptive Incremental End-to-End ML Framework with Automated Feature Engineering

## Project Overview
AdaptoML_UX is an innovative machine learning framework that offers an intuitive graphical user interface. It's designed to streamline and automate key stages in the machine learning workflow, such as feature extraction, selection, and model adaptation, making it more accessible for users with varying levels of expertise in machine learning.

## Installation Instructions
To get started with AdaptoML_UX, ensure you have Python 3.x installed on your system. Follow these steps:
1. Clone the repository or download the source code.
2. Navigate to the project directory and set up a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On Unix or MacOS: `source venv/bin/activate`
4. Install the required dependencies: `pip install -r requirements.txt`

## Usage Guide
Run `MainGUI.py` to start the application. The GUI is self-explanatory and user-friendly, guiding you through various steps of machine learning model development:
- Select data files for training and testing.
- Choose methods for feature extraction, selection, and model adaptation.
- Customize classifiers or regressors as per your requirement.
- View and analyze the results within the interface.

## Code Structure
- `MainGUI.py`: Core file for the application's graphical interface.
- `ComputeOutput.py`: Manages the computation and processing of models.
- `Classifiers_Regressors.py`: Implements various machine learning models.
- Other files like `FeatureExtraction.py`, `FeatureSelection.py`, and `Imputation.py` handle respective functionalities in the ML pipeline.
- `Utilities.py`: Provides supporting functions used throughout the application.

## License Information
[Specify License Here]

## Contact Information
For support, queries, or contributions, please reach out to amr
