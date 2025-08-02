import base64
import sys

def encode_image_to_base64(image_path):
    """Reads an image and prints its Base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print(encoded_string)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        encode_image_to_base64(sys.argv[1])
    else:
        print("Usage: python encode_image.py <path_to_image>", file=sys.stderr)
