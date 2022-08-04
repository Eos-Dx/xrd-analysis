"""
Script for generating test images
"""
import numpy as np

# Set input directory
INPUT_DIR="input"

# Set up test image
image = np.zeros((256,256), dtype=np.uint16)
center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

# Set the filename
filename = "0.txt"
# Set the full output path
fullpath = os.path.join("input", filename)
# Save the image to file
np.savetxt(fullpath, image, fmt="%d")
