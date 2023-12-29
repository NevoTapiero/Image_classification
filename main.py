import os
import numpy as np
import cv2  # Importing OpenCV for image processing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight



def load_images(input_dir, categories, expected_shape):
    """
    Load images from the specified directory, resize them to the expected shape,
    and normalize the pixel values.

    Args:
    input_dir (str): Directory containing the image data.
    categories (list): List of category names (subdirectory names).
    expected_shape (tuple): The expected shape of the images (height, width, channels).

    Returns:
    tuple: A tuple containing the array of image data and labels.
    """
    data = []
    labels = []
    for category_index, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        for file in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)  # Read the image
                img = cv2.resize(img, expected_shape[:2])  # Resize the image
                if img.shape != expected_shape:
                    print(f"Skipping file with unexpected shape: {file}")
                    continue  # Skip this image if it doesn't match the expected shape

                img = img.flatten() / 255.0  # Normalize pixel values to [0, 1]
                data.append(img)
                labels.append(category_index)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return np.array(data), np.array(labels)


def train_classifier(x_train, y_train, labels, class_weights):
    """
    Train an SVM classifier using the provided training data and class weights.
    Perform grid search to find the optimal hyperparameters.

    Args:
    x_train (array): Training data features.
    y_train (array): Training data labels.
    labels (array): Unique labels of the data.
    class_weights (array): Weights for each class to handle imbalanced data.

    Returns:
    trained model: The best estimator from the grid search.
    """
    # Convert class weights to dictionary format as required by SVC
    weight_dict = {i: weight for i, weight in zip(np.unique(labels), class_weights)}

    classifier = SVC(class_weight=weight_dict)
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(classifier, parameters, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

def load_and_preprocess_image(image_path, expected_shape):
    """
    Load and preprocess a single image for classification.

    Args:
    image_path (str): Path to the image to be processed.
    expected_shape (tuple): The expected shape of the images (height, width, channels).

    Returns:
    numpy array: Preprocessed image suitable for classification.
    """
    img = cv2.imread(image_path)  # Read the image
    img = cv2.resize(img, expected_shape[:2])  # Resize the image
    img = img.flatten() / 255.0  # Normalize pixel values to [0, 1]
    return np.array([img])  # Return the image as a batch of size 1

def main():
    """
    Main function to execute the script. It loads the data, splits it,
    computes class weights, trains the classifier, and prints performance metrics.
    """
    # Configure paths and parameters for multi-class
    input_dir = r"C:\Users\nevot\Desktop\Data1"  # Make sure this directory has subdirectories for each class
    categories = ['Corn__common_rust', 'Corn__healthy', 'Corn__gray_leaf_spot', 'Corn__northern_leaf_blight']
    expected_shape = (15, 15, 3)

    # Load and preprocess data
    data, labels = load_images(input_dir, categories, expected_shape)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Compute class weights for handling imbalanced classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

    # Train the classifier and find the best model parameters
    best_estimator = train_classifier(x_train, y_train, labels, class_weights)

    # Evaluate the classifier's performance on the test data
    y_prediction = best_estimator.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)

    # Print the accuracy and a detailed classification report
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_prediction))

    # After training and evaluating the model, use it for prediction
    new_image_path = r"C:\Users\nevot\Desktop\Data1\Corn__gray_leaf_spot\0a403456-5c5e-4aad-aa89-a118175c6ddd___RS_GLSp 4501.JPG"
    new_image = load_and_preprocess_image(new_image_path, expected_shape)
    prediction = best_estimator.predict(new_image)

    # Convert the numerical prediction back to a category name
    predicted_category = categories[prediction[0]]
    print(f"The image is classified as: {predicted_category}")


if __name__ == '__main__':
    main()