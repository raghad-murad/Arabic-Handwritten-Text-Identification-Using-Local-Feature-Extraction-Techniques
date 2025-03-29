'''
- Assignment Title: Arabic Handwritten Text Identification Using Local Feature Extraction Techniques

- Assignment Description: This assignment involves classifying images from the AHAWP (Arabic Handwritten Automatic Word Processing) dataset 
  using SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) algorithms. The task includes data reading 
  and analysis, dataset splitting, feature extraction using SIFT and ORB, building a Bag of Visual Words (BoVW) model, 
  and implementing cross-validation for performance evaluation.

- Student Information:
    - Name: Raghad Murad Buzia
    - ID: 1212214
    - Section: 2
'''

########################################################################################################################################
#                                                      Import Important Libraries                                                      #
########################################################################################################################################

'''
This section imports essential libraries for data handling, visualization, and image processing. Key libraries include:
- `numpy`: For numerical operations.
- `random`: To generate random numbers.
- `seaborn`: For statistical data visualization.
- `matplotlib.pyplot`: For plotting.
- `warnings`: To control warnings display.
- `os`: For operating system interactions.
- `cv2`: For image processing using OpenCV.
- `time`: To measure execution time.
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import time
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# Settings for reproducibility and aesthetics are applied:
np.random.seed(0)                                                          # Set the random seed for reproducibility
warnings.filterwarnings('ignore')                                          # Ignore all warnings
sns.set_style("whitegrid", {'axes.grid' : False})                          # Set the seaborn style to whitegrid and disable grid on axes

########################################################################################################################################
#                                                         Loading and Analyzing Data                                                   #
########################################################################################################################################

# Function to load dataset:
"""
    - Declare a function `load_images_dataset(dataset_directory_path)` wich take the images dataset (AHAWP) path and read images from it, 
      organize them with their user IDs, and process them. Also, extract basic information about the dataset such as number of images 
      per user and number of words.
    - Args:
        data_path (str): Path to the root directory containing user folders.
    - Returns:
        images_dataset (list): List of dictionaries containing user IDs and processed images. 
        stats (dict): Basic statistics about the dataset. 
        failed_to_load (list): List of images that failed to load.
"""
def load_images_dataset(dataset_directory_path):
    
    images = []
    failed_to_load_images = []
    stats = {"total_images": 0, "users": {}, "total_users": 0, "words_per_user": {}}

    users = os.listdir(dataset_directory_path)
    
    for user in users:

        user_path = os.path.join(dataset_directory_path, user)

        if os.path.isdir(user_path):
            stats["total_users"] += 1
            stats["users"][user] = 0

        for image_name in os.listdir(user_path):
                image_path = os.path.join(user_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    processed_image = process_image(image)
                    images.append({"user_id": user, "image": processed_image})
                    stats["total_images"] += 1
                    stats["users"][user] += 1

                    # Extract the word label (assuming format: word_userXXXX_imgXX.png)
                    word_label = image_name.split("_")[0]
                    if user not in stats["words_per_user"]:
                        stats["words_per_user"][user] = {}
                    if word_label not in stats["words_per_user"][user]:
                        stats["words_per_user"][user][word_label] = 0
                    stats["words_per_user"][user][word_label] += 1
                else:
                    failed_to_load_images.append({"user_id": user, "image_path": image_path})
    
    return images, stats, failed_to_load_images

# Function to process image:
"""
    - Declare a function `process_image(image)` to do a preprocess the input image by resizing and normalizing.
    - Args:
        image (numpy.ndarray): Input image.
    - Returns:
        processed_image (numpy.ndarray): Processed image.
"""
def process_image(image):
    # Parameters for resizing and normalization: 
    fixed_size = (296, 123) 
    # Resize the image:
    resized_image = cv2.resize(image, fixed_size)
    # Normalize the image:
    normalized_image = np.array(resized_image, dtype=np.float32) / 255.0 
    image_uint8 = (normalized_image * 255).astype('uint8') 
    return image_uint8

# Function to analyze and visualize dataset:
"""
    - Declare a function `analyze_and_visualize(stats, failed_to_load)` to analyze and visualize dataset statistics and handle failed images.
    - Args:
        stats (dict): Dataset statistics.
        failed_to_load (list): List of failed images.
"""
def analyze_and_visualize(stats, failed_to_load):
    
    # Convert stats to pandas DataFrame for better visualization
    users_data = {
        "user_id": list(stats["users"].keys()),
        "num_images": list(stats["users"].values())
    }
    users_df = pd.DataFrame(users_data)

    # Calculate mean and standard deviation of word count per user
    word_counts = [len(words) for words in stats["words_per_user"].values()]
    mean_word_count = np.mean(word_counts)
    std_word_count = np.std(word_counts)

    print("\n  - Summary Statistics:")
    print(tabulate(users_df.describe(), headers='keys', tablefmt='pretty'))
    print("\n  - Mean word count per user: {:.2f}".format(mean_word_count))
    print("\n  - Standard deviation of word count per user: {:.2f}".format(std_word_count))

    # Plot the number of images per user
    plt.figure(figsize=(10, 6))
    sns.barplot(x="user_id", y="num_images", data=users_df, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Number of Images per User")
    plt.xlabel("User ID")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # Save failed image log
    if failed_to_load:
        failed_df = pd.DataFrame(failed_to_load)
        failed_df.to_csv("failed_images_log.csv", index=False)
        print("\n  - Failed image log saved to 'failed_images_log.csv'.")

########################################################################################################################################
#                                                        Apply SIFT and ORB Algorithm                                                  #
########################################################################################################################################

# Function to extract features using SIFT algorithm:
"""
    - Declare a function `extract_features_using_sift(image)` to extract local features from images using SIFT.
    - Args:
        images (list): List of dictionaries containing user IDs and processed images.
    - Returns:
        features (list): List of features for each image.
        keypoints_list (list): List of keypoints for each image.
        descriptors_list (list): List of descriptors for each image.
"""
def extract_features_using_sift(images):
    sift = cv2.SIFT_create()
    features = []
    keypoints_list = []
    descriptors_list = []
    start_time = time.time()

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image["image"], None)
        features.append({"keypoints": keypoints, "descriptors": descriptors})
        keypoints_list.append(len(keypoints) if keypoints else 0)
        descriptors_list.append(descriptors if descriptors is not None else np.array([]))

    end_time = time.time()
    extraction_time = end_time - start_time
    # print(f"Feature extraction time (SIFT): {extraction_time:.4f} seconds")
    return features, keypoints_list, descriptors_list, extraction_time

# Function to extract features using ORB algorithm:
"""
    - Declare a function `extract_features_using_orb(images)` to extract local features from an images using ORB.
    - Args:
        images (list): List of dictionaries containing user IDs and processed images.
    - Returns:
        features (list): List of features for each image.
        keypoints_list (list): List of keypoints for each image.
        descriptors_list (list): List of descriptors for each image.
"""
def extract_features_using_orb(images):
    orb = cv2.ORB_create()
    features = []
    keypoints_list = []
    descriptors_list = []
    start_time = time.time()

    for image in images:
        keypoints, descriptors = orb.detectAndCompute(image["image"], None)
        features.append({"keypoints": keypoints, "descriptors": descriptors})
        keypoints_list.append(len(keypoints) if keypoints else 0)
        descriptors_list.append(descriptors if descriptors is not None else np.array([]))

    end_time = time.time()
    extraction_time = end_time - start_time
    # print(f"Feature extraction time (ORB): {extraction_time:.4f} seconds")
    return features, keypoints_list, descriptors_list, extraction_time

########################################################################################################################################
#                                                        Build Bag of Visual Words Model                                              #
########################################################################################################################################

# Function to build Bag of Visual Words model:
'''
    - Declare a function `build_bovw_model(descriptors_list, num_clusters=100)` to build the Bag of Visual Words model using K-Means.
    - Args:
        descriptors_list (list): List of descriptors extracted from training images.
        num_clusters (int): Number of clusters (Visual Words).
    - Returns:
        kmeans_model (KMeans): Trained K-Means model.
'''
def build_bovw_model(descriptors_list, num_clusters=100):
    all_descriptors = np.vstack([desc for desc in descriptors_list if desc is not None])
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans_model.fit(all_descriptors)
    return kmeans_model

# Function to create histograms for images:
'''
    - Declare a function `create_histograms(kmeans_model, descriptors_list, num_clusters=100)` to create histograms based on the Bag of Visual Words model.
    - Args:
        kmeans_model (KMeans): Trained K-Means model.
        descriptors_list (list): List of descriptors for images.
        num_clusters (int): Number of clusters (Visual Words).
    - Returns:
        histograms (list): Histogram representations for each image.
'''
def create_histograms(kmeans_model, descriptors_list, num_clusters=100):
    start_time = time.time()
    histograms = []
    for descriptors in descriptors_list:
        histogram = np.zeros(num_clusters)
        if descriptors is not None and descriptors.size > 0:
            cluster_assignments = kmeans_model.predict(descriptors)
            for cluster in cluster_assignments:
                histogram[cluster] += 1
        histograms.append(histogram)
    end_time = time.time()
    matching_time = end_time - start_time
    # print(f"Matching time: {matching_time:.4f} seconds")
    return histograms, matching_time

########################################################################################################################################
#                                                        Evaluate Model Performance                                                   #
########################################################################################################################################

# Function to evaluate performance:
"""
    - Declare a function `evaluate_performance(histograms, labels, num_folds=5)` to evaluate model performance using cross-validation.
    - Args:
        histograms (list): Histogram representations of images.
        labels (list): True labels for each image.
        num_folds (int): Number of folds for cross-validation.
    - Returns:
        metrics (dict): Accuracy, time efficiency, and robustness metrics.
"""
def evaluate_performance(histograms, labels, num_folds=5, model_type="KNN"):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    accuracies = []
    times = []

    for train_index, test_index in skf.split(histograms, labels):
        X_train, X_test = np.array(histograms)[train_index], np.array(histograms)[test_index]
        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # Choose the model based on its type
        if model_type == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=0)
        elif model_type == "SVM":
            model = SVC(kernel='linear', probability=True, random_state=0)
        else:
            raise ValueError("Invalid model type. Use 'KNN', 'RandomForest', or 'SVM'.")

        start_time = time.time()
        model.fit(X_train.tolist(), y_train)
        predictions = model.predict(X_test.tolist())
        end_time = time.time()

        accuracies.append(accuracy_score(y_test, predictions))
        times.append(end_time - start_time)

    return {
        "accuracy": np.mean(accuracies),
        "time_efficiency": np.mean(times),
        "robustness": np.std(accuracies),
        "model": model
    }

########################################################################################################################################
#                                                        User Input and Classification                                                #
########################################################################################################################################

# Function to classify a user-input image:
'''
    - Declare a function `classify_user_input(image_path, kmeans_model, knn_model, num_clusters)` to classify a user-input image.
    - Args:
        image_path (str): Path to the user-input image.
        kmeans_model (KMeans): Trained Bag of Visual Words model.
        knn_model (KNeighborsClassifier): Trained KNN classifier.
        num_clusters (int): Number of clusters (Visual Words).
    - Returns:
        result (str): Predicted user ID.
'''
def classify_user_image(image_path, kmeans_model, model, num_clusters, algorithm="SIFT"):
    # Read and process the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = process_image(image)

    # Extract features
    if algorithm == "SIFT":
        feature_extractor = cv2.SIFT_create()
    elif algorithm == "ORB":
        feature_extractor = cv2.ORB_create()
    else:
        raise ValueError("Invalid algorithm specified. Use 'SIFT' or 'ORB'.")

    keypoints, descriptors = feature_extractor.detectAndCompute(processed_image, None)

    # Create a histogram for the input image
    histogram = np.zeros(num_clusters)
    if descriptors is not None and len(descriptors) > 0:
        cluster_assignments = kmeans_model.predict(descriptors)
        for cluster in cluster_assignments:
            histogram[cluster] += 1

    # Reshape the histogram for prediction
    histogram = histogram.reshape(1, -1)
    predicted_user_id = model.predict(histogram)[0]

    return predicted_user_id

########################################################################################################################################
#                                                                Main Program                                                          #
########################################################################################################################################

def plot_accuracy_comparison(sift_metrics, orb_metrics):
    models = ['KNN', 'Random Forest', 'SVM']
    sift_accuracies = [sift_metrics['knn']['accuracy'], sift_metrics['rf']['accuracy'], sift_metrics['svm']['accuracy']]
    orb_accuracies = [orb_metrics['knn']['accuracy'], orb_metrics['rf']['accuracy'], orb_metrics['svm']['accuracy']]

    x = range(len(models))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, sift_accuracies, width=0.4, label='SIFT', align='center')
    plt.bar([i + 0.4 for i in x], orb_accuracies, width=0.4, label='ORB', align='center')
    plt.xticks([i + 0.2 for i in x], models)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: SIFT vs ORB')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_efficiency(sift_extraction_time, sift_matching_time, orb_extraction_time, orb_matching_time):
    stages = ['Feature Extraction', 'Matching']
    sift_times = [sift_extraction_time, sift_matching_time]
    orb_times = [orb_extraction_time, orb_matching_time]
    
    x = range(len(stages))
    
    plt.figure(figsize=(10, 6))
    plt.bar(x, sift_times, width=0.4, label='SIFT', align='center')
    plt.bar([i + 0.4 for i in x], orb_times, width=0.4, label='ORB', align='center')
    plt.xticks([i + 0.2 for i in x], stages)
    plt.xlabel('Processing Stages')
    plt.ylabel('Time (seconds)')
    plt.title('Time Efficiency: SIFT vs ORB')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():

    # Dataset folder path `Path to the main directory which containing user subfolders`:
    dataset_directory_path = 'isolated_words_per_user'

    ##################################################   Loading and Analyzing Data   ##################################################
    
    # Load dataset:
    print("\nImages Dataset Start Loading....")
    images, stats, failed_to_load_images = load_images_dataset(dataset_directory_path)
    print("\nEnd Loading Images!!")

    # Display the number of loaded images and faild loaded images:
    print("\nSummary of Image Loading:")
    print("\n  - Total images loaded: {}".format(stats['total_images']))
    print("\n  - Total users: {}".format(stats['total_users']))
    print("\n  - Users stats: {}".format(stats['users']))
    if failed_to_load_images is None:
        print("\n  - All Images Loaded Successfuly!!")
    else:
        print("\n  - There is Some Images Faild to Load! (Note: Total images failed to load: {}).".format(len(failed_to_load_images)))
    
    analyze_and_visualize(stats, failed_to_load_images)

    # Prepare labels for evaluation (Assume each image has a label "user_id")
    labels = [image["user_id"] for image in images]

    #######################################################   SIFT Processing   ########################################################
    
    print("\nStarting Extracting Features using SIFT...")
    sift_features, sift_keypoints, sift_descriptors, sift_extraction_time = extract_features_using_sift(images)
    print("\nEnd Extracting Features using SIFT...")

    print("\nBuilding Bag of Visual Words Model for SIFT...")
    sift_kmeans = build_bovw_model(sift_descriptors, num_clusters=300)
    sift_histograms, sift_matching_time = create_histograms(sift_kmeans, sift_descriptors, num_clusters=300)

    print("\n - SIFT Feature Extraction Time: {:.4f} seconds".format(sift_extraction_time))
    print(" - SIFT Matching Time: {:.4f} seconds".format(sift_matching_time))
    print(" - SIFT Total Time: {:.4f} seconds".format((sift_extraction_time + sift_matching_time)))

    # Evaluate SIFT with KNN
    print("\nEvaluating Model Performance using SIFT with KNN...")
    sift_knn_metrics = evaluate_performance(sift_histograms, labels, model_type="KNN")
    print("\nSIFT KNN Metrics:")
    print(tabulate([sift_knn_metrics], headers="keys", tablefmt="pretty"))

    # Evaluate SIFT with Random Forest
    print("\nEvaluating Model Performance using SIFT with Random Forest...")
    sift_rf_metrics = evaluate_performance(sift_histograms, labels, model_type="RandomForest")
    print("\nSIFT Random Forest Metrics:")
    print(tabulate([sift_rf_metrics], headers="keys", tablefmt="pretty"))

    # Evaluate SIFT with SVM
    print("\nEvaluating Model Performance using SIFT with SVM...")
    sift_svm_metrics = evaluate_performance(sift_histograms, labels, model_type="SVM")
    print("\nSIFT SVM Metrics:")
    print(tabulate([sift_svm_metrics], headers="keys", tablefmt="pretty"))

    ########################################################   ORB Processing   ########################################################
    
    print("\nStarting Extracting Features using ORB...")
    orb_features, orb_keypoints, orb_descriptors, orb_extraction_time = extract_features_using_orb(images)
    print("\nEnd Extracting Features using ORB...")

    print("\nBuilding Bag of Visual Words Model for ORB...")
    orb_kmeans = build_bovw_model(orb_descriptors, num_clusters=300)
    orb_histograms, orb_matching_time = create_histograms(orb_kmeans, orb_descriptors, num_clusters=300)

    print("\n - ORB Feature Extraction Time: {:.4f} seconds".format(orb_extraction_time))
    print(" - ORB Matching Time: {:.4f} seconds".format(orb_matching_time))
    print(" - ORB Total Time: {:.4f} seconds".format((orb_extraction_time + orb_matching_time)))

    # Evaluate ORB with KNN
    print("\nEvaluating Model Performance using ORB with KNN...")
    orb_knn_metrics = evaluate_performance(orb_histograms, labels, model_type="KNN")
    print("\nORB KNN Metrics:")
    print(tabulate([orb_knn_metrics], headers="keys", tablefmt="pretty"))

    # Evaluate ORB with Random Forest
    print("\nEvaluating Model Performance using ORB with Random Forest...")
    orb_rf_metrics = evaluate_performance(orb_histograms, labels, model_type="RandomForest")
    print("\nORB Random Forest Metrics:")
    print(tabulate([orb_rf_metrics], headers="keys", tablefmt="pretty"))

    # Evaluate ORB with SVM
    print("\nEvaluating Model Performance using ORB with SVM...")
    orb_svm_metrics = evaluate_performance(orb_histograms, labels, model_type="SVM")
    print("\nORB SVM Metrics:")
    print(tabulate([orb_svm_metrics], headers="keys", tablefmt="pretty"))

    ########################################################   Visualization   ########################################################
    
    # Plot Accuracy Comparison
    plot_accuracy_comparison(
        {'knn': sift_knn_metrics, 'rf': sift_rf_metrics, 'svm': sift_svm_metrics},
        {'knn': orb_knn_metrics, 'rf': orb_rf_metrics, 'svm': orb_svm_metrics}
    )

    # Plot Time Efficiency
    plot_time_efficiency(
        sift_extraction_time, sift_matching_time,
        orb_extraction_time, orb_matching_time
    )

    #########################################################   Save Models   ##########################################################

    with open('sift_kmeans_model.pkl', 'wb') as f:
        pickle.dump(sift_kmeans, f)
    with open('knn_model_sift.pkl', 'wb') as f:
        pickle.dump(sift_knn_metrics["model"], f)
    with open('rf_model_sift.pkl', 'wb') as f:
        pickle.dump(sift_rf_metrics["model"], f)
    with open('svm_model_sift.pkl', 'wb') as f:
        pickle.dump(sift_svm_metrics["model"], f)

    with open('orb_kmeans_model.pkl', 'wb') as f:
        pickle.dump(orb_kmeans, f)
    with open('knn_model_orb.pkl', 'wb') as f:
        pickle.dump(orb_knn_metrics["model"], f)
    with open('rf_model_orb.pkl', 'wb') as f:
        pickle.dump(orb_rf_metrics["model"], f)
    with open('svm_model_orb.pkl', 'wb') as f:
        pickle.dump(orb_svm_metrics["model"], f)

    print("\nModels saved successfully!")

    #####################################################   Classify User Input   ######################################################
    
    print("\nClassify User Input:")
    while True:
        user_input = input("\n - Enter the path of the image to classify (or e to exit): ")
        if user_input == 'e' or user_input == 'E':
            print("\nExiting The Program...\n")
            break
        else:
            user_input_image_path = user_input
            
            # SIFT with KNN
            predicted_user_sift_knn = classify_user_image(user_input_image_path, sift_kmeans, sift_knn_metrics["model"], num_clusters=300, algorithm="SIFT")
            print("   # Predicted User ID using SIFT with KNN: {}".format(predicted_user_sift_knn))

            # SIFT with Random Forest
            predicted_user_sift_random_forest = classify_user_image(user_input_image_path, sift_kmeans, sift_rf_metrics["model"], num_clusters=300, algorithm="SIFT")
            print("   # Predicted User ID using SIFT with Random Forest: {}".format(predicted_user_sift_random_forest))

            # SIFT with SVM
            predicted_user_sift_svm = classify_user_image(user_input_image_path, sift_kmeans, sift_svm_metrics["model"], num_clusters=300, algorithm="SIFT")
            print("   # Predicted User ID using SIFT with SVM: {}".format(predicted_user_sift_svm))

            # ORB with KNN
            predicted_user_orb_knn = classify_user_image(user_input_image_path, orb_kmeans, orb_knn_metrics["model"], num_clusters=300, algorithm="ORB")
            print("   # Predicted User ID using ORB with KNN: {}".format(predicted_user_orb_knn))

            # ORB with Random Forest
            predicted_user_orb_random_forest = classify_user_image(user_input_image_path, orb_kmeans, orb_rf_metrics["model"], num_clusters=300, algorithm="ORB")
            print("   # Predicted User ID using ORB with Random Forest: {}".format(predicted_user_orb_random_forest))

            # ORB with SVM
            predicted_user_orb_svm = classify_user_image(user_input_image_path, orb_kmeans, orb_svm_metrics["model"], num_clusters=300, algorithm="ORB")
            print("   # Predicted User ID using ORB with SVM: {}".format(predicted_user_orb_svm))

########################################################################################################################################
#                                                                Run The Code                                                          #
########################################################################################################################################

if __name__ == "__main__":
    main()