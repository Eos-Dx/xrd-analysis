"""
Script for generating test images
"""
import numpy as np

# Set input directory
INPUT_DIR="input"

# Set up test image parameters
h=256
w=256
min_val=0
max_val=255
num = 2

# Generate test images
for count in range(num):
    # Generate random test image with integer values
    image = np.random.randint(0,255,(256,256), dtype=np.uint32)
    # Set the filename
    filename = str(count) + ".txt"
    # Set the full output path
    fullpath = os.path.join("input", filename)
    # Save the image to file
    np.savetxt(fullpath, image, fmt="%d")
