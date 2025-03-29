# Computer Vision Assignment 2: Arabic Handwritten Text Identification Using Local Feature Extraction Techniques

This repository contains the implementation and report for **Assignment 2** in the Computer Vision course. The assignment focuses on identifying Arabic handwritten text using local feature extraction techniques such as SIFT and ORB. The goal is to classify images from the **AHAWP (Arabic Handwritten Automatic Word Processing) dataset**, extract features, build classifiers, and evaluate their performance using metrics like accuracy, precision, recall, and F1-score.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Implementation Details](#implementation-details)
- [Files in the Repository](#files-in-the-repository)
- [How to Run the Project](#how-to-run-the-project)
- [Results and Visualizations](#results-and-visualizations)
- [Contributions](#contributions)
- [License](#license)

---

## 🌟 Overview

The objective of this assignment is to:

1. **Load and Preprocess the Dataset**:
   - Load images from the AHAWP dataset.
   - Preprocess images to enhance feature extraction.

2. **Feature Extraction**:
   - Extract local features using:
     - **SIFT (Scale-Invariant Feature Transform)**.
     - **ORB (Oriented FAST and Rotated BRIEF)**.

3. **Build Bag of Visual Words (BoVW) Model**:
   - Cluster extracted features using K-means clustering.
   - Create histograms representing the distribution of visual words.

4. **Train Classifiers**:
   - Train multiple classifiers using extracted features:
     - k-Nearest Neighbors (kNN).
     - Random Forest.
     - Support Vector Machine (SVM).

5. **Evaluate Performance**:
   - Use cross-validation to evaluate classifier performance.
   - Report metrics such as accuracy, precision, recall, and F1-score.

6. **Compare Algorithms**:
   - Compare the performance of SIFT and ORB features across different classifiers.
   - Analyze the impact of feature extraction techniques on classification accuracy.

---

## 📊 Dataset

The dataset used in this project is the **AHAWP (Arabic Handwritten Automatic Word Processing) dataset**, which contains images of Arabic handwritten text. The dataset includes:
- **Images**: Various samples of Arabic handwritten words.
- **Labels**: Corresponding labels indicating the class of each image.

The dataset is split into training and testing sets for model evaluation.

---

## 🛠️ Implementation Details

The project is implemented using Python with the following libraries:
- **OpenCV**: For image processing and feature extraction (SIFT, ORB).
- **NumPy**: For numerical operations.
- **scikit-learn**: For building and evaluating classifiers.
- **Matplotlib & Seaborn**: For data visualization.

### Key Steps in the Implementation

1. **Image Loading and Preprocessing**:
   - Load images from the dataset.
   - Convert images to grayscale and apply preprocessing steps (e.g., resizing, normalization).

2. **Feature Extraction**:
   - Use OpenCV's `cv2.SIFT_create()` and `cv2.ORB_create()` to extract local features.
   - Apply K-means clustering to create a Bag of Visual Words (BoVW) representation.

3. **Classifier Training**:
   - Train classifiers using extracted features:
     - **k-Nearest Neighbors (kNN)**.
     - **Random Forest**.
     - **Support Vector Machine (SVM)**.

4. **Evaluation Metrics**:
   - Calculate accuracy, precision, recall, and F1-score for each classifier.
   - Perform cross-validation to ensure robust evaluation.

5. **Visualization**:
   - Generate plots comparing classifier performance.
   - Display feature matching examples using SIFT and ORB.

---

## 📁 Files in the Repository

The repository contains the following files:

### Main Files
- **`Assignment_Two_Code_1212214.py`**: The main Python script implementing the feature extraction pipeline and classifier training.
- **`AssignmentTwo_Report_RaghadMuradBuzia_1212214.pdf`**: Detailed report explaining the methodology, results, and analysis.

### Data Files
- **`isolated_words_per_user`**: Folder containing the AHAWP dataset images.

### Model Files
- **`.pkl` Files**: Serialized models for trained classifiers and BoVW models:
  - `knn_model_orb.pkl`, `knn_model_sift.pkl`: kNN models trained using ORB and SIFT features.
  - `rf_model_orb.pkl`, `rf_model_sift.pkl`: Random Forest models trained using ORB and SIFT features.
  - `svm_model_orb.pkl`, `svm_model_sift.pkl`: SVM models trained using ORB and SIFT features.
  - `orb_kmeans_model.pkl`, `sift_kmeans_model.pkl`: K-means models for BoVW using ORB and SIFT features.

### Visualization Files
- **PNG Files**: Visualizations generated during the analysis:
  - `Accuracy of Classifiers - Accuracy Comparison.png`: Comparison of classifier accuracies.
  - `Dataset Visualization - Number of Images per Class.png`: Distribution of images across classes.
  - `Feature Extraction and Matching Time - Time Comparison.png`: Comparison of feature extraction and matching times.
  - `Run Example - Part 1.png` to `Run Example - Part 6.png`: Step-by-step execution examples.

---

## 🚀 How to Run the Project

### Prerequisites
- **Python**: Ensure Python is installed on your machine.
- **Libraries**: Install the required libraries using `pip`:
  ```bash
  pip install opencv-python numpy scikit-learn matplotlib seaborn
  ```

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/raghad-murad/ComputerVisionAssignment2.git
   ```

2. **Navigate to the Directory**
   ```bash
   cd ComputerVisionAssignment2
   ```

3. **Run the Main Script**
   ```bash
   python Assignment_Two_Code_1212214.py
   ```

4. **View Results**
   - Output PNG files will be generated in the directory.
   - Trained models will be saved as `.pkl` files.

---

## 📊 Results and Visualizations

The project generates various outputs, including:
- **PNG Files**: Visualizations of classifier performance, feature extraction times, and dataset distributions.
- **Performance Metrics**: Accuracy, precision, recall, and F1-score for each classifier.
- **Trained Models**: Serialized models for future use.

Example visualizations include:
- Bar charts comparing classifier accuracies.
- Line plots showing feature extraction and matching times.
- Heatmaps illustrating confusion matrices.

---

## 🤝 Contributions

If you'd like to contribute to this repository, feel free to:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed explanation of your changes.

---

## 📧 Contact

If you have any questions or suggestions, feel free to reach out!

- **Email:** raghadmbuzia@gmail.com
- **LinkedIn:** [in/raghad-murad](http://linkedin.com/in/raghad-murad-02690433a)

---

# Thank you for checking out this project! 🚀