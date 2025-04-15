import tensorflow as tf
import numpy as np
import cv2
import os

# Load the saved model
model_path = "best_model"
model = tf.saved_model.load(model_path)

# Assuming the model is for keypoint detection, check the available signature
signature = list(model.signatures.keys())
print(f"Model signatures: {signature}")
infer = model.signatures["serving_default"]  # The default signature key

# Print model's input signature to confirm the expected dimensions
print(infer.structured_input_signature)

# Inspect the model's output keys
output_keys = list(infer.structured_outputs.keys())
print(f"Model output keys: {output_keys}")

# Function to preprocess an image
def preprocess_image(image_path, target_size=(240, 240)):  # Update to the model's expected input size
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, target_size)  # Resize to the model's input size
    image_resized = image_resized / 255.0  # Normalize if required
    # Ensure the image has 3 channels (RGB)
    if image_resized.shape[-1] != 3:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    return image_resized

# Function to infer on a batch of images
def infer_images(image_paths):
    # Preprocess all images
    images = np.array([preprocess_image(image_path) for image_path in image_paths])
    
    # Add a batch dimension and convert the images to a tensor
    input_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

    # Perform inference (run on CPU if necessary)
    with tf.device('/CPU:0'):  # Change to '/GPU:0' if you want to use GPU
        predictions = infer(input_tensor)

    # Update this with the correct output key (found in the print statements)
    output_key = output_keys[0]  # Assuming the first output key is the one we need
    output = predictions[output_key]  # Access the actual predictions
    
    # Return the predictions as a numpy array
    return output.numpy()

# Function to save annotated image with keypoint predictions
def save_annotated_image(image_path, keypoints, output_dir="predicted_images"):
    # Read the original image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Iterate through the keypoints and annotate the image
    for (x, y, visibility) in keypoints:
        # Check if the keypoint is visible (visibility should be 0 or 1)
        if visibility > 0.5:  # Assuming visibility is between 0 and 1, and threshold is 0.5
            # Rescale the keypoints if they are normalized (assuming they are in [0, 1])
            x_rescaled = int(x * width)  # Rescale x to image width
            y_rescaled = int(y * height)  # Rescale y to image height
            
            # Draw a circle for each visible keypoint
            cv2.circle(image, (x_rescaled, y_rescaled), 5, (0, 255, 0), -1)  # Green circle for keypoint
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the original file name and save with .jpg extension in the output directory
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))  # Keeps the same file name
    if not output_image_path.endswith(".jpg"):
        output_image_path = output_image_path.rsplit(".", 1)[0] + ".jpg"  # Ensure it ends with .jpg

    # Save the annotated image
    cv2.imwrite(output_image_path, image)

    return output_image_path

# List of image paths in your dataset
image_folder = "/home/weijunl/Football-Object-Detection/datasets/dataset/train/images"
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]

# Run inference on the batch of images
results = infer_images(image_paths)

# Check the shape of the results (predicted keypoints)
print(f"Results shape: {results.shape}")

# Assuming the results are keypoints (x, y, visibility), loop through and save annotated images
output_dir = "predicted_images"  # Directory to save the annotated images
for image_path, result in zip(image_paths, results):
    # Check the result structure before using it
    print(f"Result for {image_path}: {result}")
    
    # Check if the result is a scalar or array
    if result.ndim == 1:  # If the result is a flat array
        result = result.reshape(-1, 3)  # Reshape it to a 2D array (x, y, visibility)
    
    # Rescale keypoints if they are normalized (ensure correct shape)
    keypoints_rescaled = [(kp[0], kp[1], kp[2]) for kp in result]  # Assuming results are normalized (between 0 and 1)
    
    # Save annotated images
    annotated_image_path = save_annotated_image(image_path, keypoints_rescaled, output_dir)
    print(f"Annotated image saved at: {annotated_image_path}")
