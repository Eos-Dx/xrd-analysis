import base64
from io import BytesIO

from PIL import Image


def show_image_from_base64(encoded_str):
    """
    Decodes the Base64-encoded string to an image and displays it using the default image viewer.

    Parameters:
        encoded_str (str): The Base64-encoded image string.
    """
    image = decode_base64_to_image(encoded_str)
    if image:
        image.show()
    else:
        print("Failed to decode and display image.")


def encode_image_to_base64(image_path):
    """
    Reads an image file in binary mode and returns its Base64 encoded string.

    Parameters:
        image_path (str): The file path to the image.

    Returns:
        str: The Base64 encoded string of the image, or None if an error occurs.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        encoded_str = base64.b64encode(image_data).decode("utf-8")
        return encoded_str
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    # Replace the string below with your actual Base64 encoded image string.
    p = "C:\\dev\\eos_play\\unzipped_data\\Ulster\\pancreas\\sheep_pancreas\\U2\\Xena_Cu\\20250313\\P1a_L4_20250218.jpg"
    encoded_image_str = encode_image_to_base64(p)
    show_image_from_base64(encoded_image_str)
