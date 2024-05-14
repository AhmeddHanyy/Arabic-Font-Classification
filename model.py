import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
from cv2 import medianBlur, threshold, cvtColor, COLOR_BGR2GRAY, getStructuringElement, MORPH_RECT, morphologyEx, \
                CHAIN_APPROX_SIMPLE, RETR_EXTERNAL, boundingRect, filter2D, warpAffine, getRotationMatrix2D, \
                THRESH_BINARY, THRESH_OTSU, SIFT_create, getGaborKernel,MORPH_CLOSE,findContours,COLOR_GRAY2BGR, minAreaRect,boxPoints,drawContours,COLOR_BGR2RGB,CV_8U,CV_32F,COLOR_RGB2GRAY,COLOR_RGB2BGR




scaler = joblib.load("minmax_scaler.pkl")
svm_model = joblib.load("final_model.pkl")
with open("dropped_features.pkl", 'rb') as file: dropped_features = pickle.load(file)

class ImagePreprocessor:
    def __init__(self, img):
        """
        Initialize the Preprocessing class with an image.
        Args:
            img (numpy.ndarray): The image to be processed, assumed to be in grayscale.
        """
        self.img = img  # the image to be processed
        self.angles = []  # list to store the angles of text orientations
    
    def fix_color(self):
        """
        Inverts the image colors if the average color of the border pixels is light.
        This is to ensure that the text is darker than the background for better processing.
        """
        # Extract the border pixels
        top = self.img[0]
        bottom = self.img[-1]
        left = self.img[:,0]
        right = self.img[:,-1]
        
        # Calculate the average color of the border pixels
        avg = np.mean([np.mean(top), np.mean(bottom), np.mean(left), np.mean(right)])
        
        # Invert the image colors if the average is light
        if avg > 128:
            self.img = 255 - self.img

    def binarize_image(self):
        """
        Applies median blurring and Otsu's thresholding to binarize the image.
        After thresholding, it calls fix_color to ensure proper contrast.
        """
        # Apply median blurring twice to reduce noise
        self.img = medianBlur(self.img, 3)
        self.img = medianBlur(self.img, 3)
        
        # Apply Otsu's thresholding
        _, self.img = threshold(self.img, 0, 255, THRESH_BINARY + THRESH_OTSU)
        
        # Ensure the text is darker than the background
        self.fix_color()

    def get_rectangle_angles(self, structuring_element_size=20, display_rectangles=False):
        """
        Detects contours in the image and computes the orientation of each contour's minimum area rectangle.
        Optionally displays these rectangles overlaid on the image.
        Args:
            structuring_element_size (int): Size of the structuring element for morphological operations.
            display_rectangles (bool): If True, display the image with rectangles drawn around detected contours.
        """
        # Apply morphological closing to make the contours more detectable
        kernel = getStructuringElement(MORPH_RECT, (structuring_element_size, structuring_element_size))
        processed = morphologyEx(self.img, MORPH_CLOSE, kernel)

        # Find contours in the processed image
        contours, _ = findContours(processed, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Optional: prepare image for rectangle visualization
        if display_rectangles: 
            color_img = cvtColor(self.img, COLOR_GRAY2BGR)

        # Process each detected contour
        for contour in contours:
            rect = minAreaRect(contour)  # get the minimum area rectangle
            angle = rect[2]  # extract the angle
            
            # Adjust angle for a consistent representation
            if rect[1][0] < rect[1][1]:
                angle -= 90
            
            self.angles.append(angle)

            # If displaying rectangles, draw them on the image
            if display_rectangles:
                box = boxPoints(rect)
                box = np.int0(box)
                drawContours(color_img, [box], 0, (0, 255, 0), 2)

        # Show the image with drawn rectangles
        if display_rectangles:
            plt.imshow(cvtColor(color_img, COLOR_BGR2RGB))
            plt.show()

    def plot_angles_histogram(self):
        """
        Plots a histogram of the angles of text orientations.
        This only works if there are angles collected in the self.angles list.
        """
        if self.angles:
            plt.figure(figsize=(10, 6))
            plt.hist(self.angles, bins=30, color='green', alpha=0.7)
            plt.title('Distribution of Text Orientations')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

    def find_best_angle(self):
        """
        Finds the most common text orientation angle from the histogram of angles.
        Returns:
            float: The average angle from the most populated bin in the histogram.
        """
        hist, bin_edges = np.histogram(self.angles, bins=30)
        max_bin_index = np.argmax(hist)
        return np.mean([bin_edges[max_bin_index], bin_edges[max_bin_index + 1]])

    def rotate_image(self, angle):
        """
        Rotates the image by a specified angle.
        Args:
            angle (float): The angle to rotate the image by, in degrees.
        """
        (h, w) = self.img.shape[:2]  # image dimensions
        center = (w / 2, h / 2)  # image center

        # Compute the rotation matrix for the rotation and the scale
        M = getRotationMatrix2D(center, angle, 1.0)
        
        # Apply the rotation to the image
        rotated_img = warpAffine(self.img, M, (w, h))
        
        # Update the image
        self.img = rotated_img

    def find_text_region(self, structuring_element_size=150):
        """
        Find the region containing the largest bounding rectangle after applying morphological closing.

        Args:
            structuring_element_size (int): Size of the structuring element for morphological closing.

        Returns:
            numpy.ndarray: The cropped region with the largest bounding rectangle.
        """
        # Apply morphological closing to connect the text regions
        kernel = getStructuringElement(MORPH_RECT, (structuring_element_size, structuring_element_size))
        closed_img = morphologyEx(self.img, MORPH_CLOSE, kernel)

        # Find contours in the closed image
        contours, _ = findContours(closed_img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest bounding rectangle
        max_area = 0
        max_rect = None
        for contour in contours:
            x, y, w, h = boundingRect(contour)
            area = w * h
            if area > max_area:
                max_area = area
                max_rect = (x, y, w, h)

        # Crop the region with the largest bounding rectangle
        if max_rect is not None:
            x, y, w, h = max_rect
            return self.img[y:y + h, x:x + w]
        else:
            return self.img  # Return the whole image if no contours are found


    def preprocess_image(self, structuring_element_size=20, display_rectangles=False):
        """
        Executes the full preprocessing pipeline on the image, which includes binarization,
        detecting text orientations, finding the best rotation angle, and rotating the image.
        Args:
            structuring_element_size (int): Size of the kernel for morphological operations.
            display_rectangles (bool): Whether to display the rectangles around detected contours.
        Returns:
            The rotated image.
        """
        self.binarize_image()
        self.get_rectangle_angles(structuring_element_size, display_rectangles)
        best_angle = self.find_best_angle()
        self.rotate_image(best_angle)
        self.img = self.find_text_region()
        self.binarize_image()
        return self.img
    


class BoVW:
    def __init__(self):
        self.feature_vectors = []

    def extract_SIFT_descriptors(self, data):
        descriptors = []
        sift = SIFT_create()
        for img in data:
            img = (255 * img).astype(np.uint8)
            if len(img.shape) == 3: 
                img = cvtColor(img, COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(img, None)
            if desc is not None:
                descriptors.append(desc)
            else:
                print("No descriptors found for an image.")
        return descriptors
    
    def extract_BoVW(self, data, k=112, kmeans=None, sift_features=None):

            if not sift_features:
                sift_features = self.extract_SIFT_descriptors(data)
            
            if not kmeans:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
                kmeans.fit(np.vstack(sift_features))
            
            # Create histograms for each image
            for desc in sift_features:
                if desc is not None and len(desc) > 0:

                    # Predict cluster assignments for the reduced descriptors using the trained k-means model
                    cluster_predictions = kmeans.predict(desc)

                    # Create a histogram of cluster assignments (visual words)
                    hist, _ = np.histogram(cluster_predictions, bins=np.arange(kmeans.n_clusters + 1), density=True)

                    self.feature_vectors.append(hist)
                else:
                    # If no descriptors were found for this image, use an empty histogram
                    self.feature_vectors.append(np.zeros(self.kmeans.n_clusters))
            
            return np.array(self.feature_vectors), kmeans








class GaborExtractor:
    def __init__(self):
        self.filters = []
        self.feature_vectors = []

    def build_gabor_filters(self, orientations, frequencies, sigmas):
        for θ in orientations:
            for frequency, σ in zip(frequencies, sigmas):
                λ = 1 / frequency  # Wavelength
                γ = 0.5  # Spatial aspect ratio
                kernel_size = int(3 * σ) if int(3 * σ) % 2 == 1 else int(3 * σ) + 1  # Ensure kernel size is odd
                zero_kernel = getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, 0, ktype=CV_32F)
                neg_kernel = getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, np.pi/2, ktype=CV_32F)
                self.filters.append([zero_kernel, neg_kernel])

    def extract_gabor_features(self, data, orientations, frequencies, sigmas):
        self.build_gabor_filters(orientations, frequencies, sigmas)
        print("Prepared Filters")
        num_images = len(data)
        for i in range(num_images):
            print(i)
            feature_vector = np.zeros(2 * (1 + len(self.filters)), dtype=np.float32)
            feature_vector_index = 0
            image_responses = []
            for zero_kernel, neg_kernel in self.filters:
                zero_filtered = filter2D(data[i], CV_8U, zero_kernel)
                neg_filtered = filter2D(data[i], CV_8U, neg_kernel)
                E = np.sqrt(zero_filtered ** 2 + neg_filtered ** 2)
                E = np.clip(E, -1e10, 1e10)  # Clamp the values to a reasonable range
                image_responses.append(E)
                E_mean = np.mean(E)
                E_std = np.std(E)
                feature_vector[feature_vector_index] = E_mean
                feature_vector[feature_vector_index + 1] = E_std
                feature_vector_index += 2

            max_response = np.max(image_responses, axis=0)
            max_response = np.clip(max_response, -1e10, 1e10)  # Clamp the values to a reasonable range
            max_response_mean = np.mean(max_response)
            max_response_std = np.std(max_response)
            feature_vector[feature_vector_index] = max_response_mean
            feature_vector[feature_vector_index + 1] = max_response_std
            self.feature_vectors.append(feature_vector)

        # Convert the list of feature vectors to a 2D numpy array
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors




class LawsExtractor:
    def __init__(self):
        self.filters = []
        self.feature_vectors = []
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        W5 = np.array([-1, 2, 0, -2, 1])
        R5 = np.array([1, -4, 6, -4, 1])
        self.masks = [L5, E5, S5, W5, R5]

    def create_laws_kernels(self):
        for i in range(5):
            for j in range(5):
                self.filters.append(np.outer(self.masks[i], self.masks[j]))
    
    def extract_laws_texture_energy_measures(self, data):
        self.create_laws_kernels()
        i = 0
        for image in data:
            print(i)
            feature_vector = []
            for kernel in self.filters:
                filtered_image = np.abs(filter2D(image, -1, kernel))
                energy = np.mean(filtered_image)
                std = np.std(filtered_image)
                feature_vector.extend([energy, std])
            self.feature_vectors.append(feature_vector)
            i += 1
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors




class FeatureExtraction:
    def __init__(self, kmeans=None, scaler=None, dropped_features=None):
        self.kmeans = kmeans
        self.scaler = scaler
        self.dropped_features = dropped_features
    
    def load_features_from_file(self, filename):
        data = np.genfromtxt(filename, delimiter=",")
        X = data[:, :-1]
        y = data[:, -1]

        return (X, y)
    
    # Remove highly correlated features
    def remove_highly_correlated_features(self, X, threshold=0.95):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_reduced = np.delete(X, to_drop, axis=1)
        return X_reduced, to_drop
    
    def extract_features(self, X=None):
        orientations = [k * np.pi / 8 for k in range(1, 9)]
        frequencies = np.linspace(0.2, 0.5, 3)
        sigmas = np.linspace(3, 1, 3)
        BoVW_extractor = BoVW()
        gabor_extractor = GaborExtractor()
        laws_extractor = LawsExtractor()
        X_BoVW, _ = BoVW_extractor.extract_BoVW(data=X, kmeans=self.kmeans)
        X_Gabor = gabor_extractor.extract_gabor_features(X, orientations, frequencies, sigmas)
        X_Laws = laws_extractor.extract_laws_texture_energy_measures(X)
        X = np.concatenate((X_BoVW, X_Gabor), axis=1)
        X = np.concatenate((X, X_Laws), axis=1)
        X = self.scaler.transform(X)
        X = np.delete(X, self.dropped_features, axis=1)
        return X
    




class ImagePreprocessorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        processed_images = []
        for img in X:
            img = cvtColor(img, COLOR_RGB2GRAY)
            image_processor = ImagePreprocessor(img)
            processed_img = image_processor.preprocess_image()
            processed_images.append((processed_img > 127).astype(np.uint8))
        return np.array(processed_images)
    





class FeatureExtractorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, kmeans, scaler, dropped_features):
        self.kmeans = kmeans
        self.scaler = scaler
        self.dropped_features = dropped_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feature_extractor = FeatureExtraction(self.kmeans, self.scaler, self.dropped_features)
        features = feature_extractor.extract_features(X)
        return features