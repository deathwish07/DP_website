from PIL import Image
import imagehash
import os

threshold = 25 # Define the threshold value for similarity detection

image_dir = "G:\My Drive\Cloud\Bike"  # Replace with the path to the directory containing the images

# Loop over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check that the file is an image
        filepath = os.path.join(image_dir, filename)
        # Load the image using Pillow
        image1 = Image.open(filepath)
        # Calculate the perceptual hash of the image using imagehash
        hash1 = imagehash.phash(image1)
        # Compare the image to all other images in the directory
        for other_filename in os.listdir(image_dir):
            if other_filename != filename and (other_filename.endswith(".jpg") or other_filename.endswith(".png")):
                other_filepath = os.path.join(image_dir, other_filename)
                # Load the other image using Pillow
                image2 = Image.open(other_filepath)
                # Calculate the perceptual hash of the other image using imagehash
                hash2 = imagehash.phash(image2)
                # If the hamming distance between the hashes is less than or equal to the threshold, delete the current image
                if hash1 - hash2 <= threshold:
                    os.remove(filepath)
                    
                    break  # Once an image is deleted, stop comparing it to other images in the directory