import numpy as np
import cv2
from .RefinedTramsmission import Refinedtransmission
from .getAtomsphericLight import getAtomsphericLight
from .getGbDarkChannel import getDarkChannel
from .getTM import getTransmission
from .sceneRadiance import sceneRadianceRGB
# Taken from: https://github.com/huabo-zhu/underwater-advancement
np.seterr(over="ignore")


def apply_udcp(image):
    """
    Applies the Underwater Dark Channel Prior (UDCP) enhancement method to the input image.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array (BGR format).

    Returns:
        numpy.ndarray: Enhanced image as a NumPy array (BGR format).
    """
    # Parameters for UDCP
    block_size = 9

    # Step 1: Get the Dark Channel
    dark_channel = getDarkChannel(image, block_size)

    # Step 2: Get the Atmospheric Light
    atmospheric_light = getAtomsphericLight(dark_channel, image)

    # Step 3: Get the Transmission Map
    transmission = getTransmission(image, atmospheric_light, block_size)

    # Step 4: Refine the Transmission Map
    refined_transmission = Refinedtransmission(transmission, image)

    # Step 5: Compute the Scene Radiance
    enhanced_image = sceneRadianceRGB(image, refined_transmission, atmospheric_light)

    return enhanced_image


# Example usage (for debugging):
# if __name__ == "__main__":
#     input_path = r'C:\Users\Siddhi Patil\Desktop\Methods\underwater-advancement\img\91_img_.png'  # Replace with your test image path
#     output_path = r'C:\Users\Siddhi Patil\Desktop\Methods\underwater-advancement\91.png'  # Replace with desired output path

#     input_image = cv2.imread(input_path)
#     if input_image is not None:
#         result = apply_udcp(input_image)
#         cv2.imwrite(output_path, result)
#         print("UDCP enhancement completed. Output saved to:", output_path)
#     else:
#         print("Error: Unable to load the input image.")
