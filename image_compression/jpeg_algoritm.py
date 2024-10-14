import cv2
import numpy as np

def compress_image(image_path, output_path, quality):
    # Read the image
    img = cv2.imread(image_path)

    # Encode the image with JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)

    # Decode the compressed image
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    # Save the compressed image as a new file
    cv2.imwrite(output_path, decoded_img)

    return decoded_img

# Example usage
original_img = cv2.imread('jpeg/sample.bmp')
compressed_img = compress_image('jpeg/sample.bmp', 'jpeg/compressed_image.jpg', quality=10)

# Display results
cv2.waitKey(0)
cv2.destroyAllWindows()
