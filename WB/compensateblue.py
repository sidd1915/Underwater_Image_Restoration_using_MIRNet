from PIL import Image, ImageStat,  ImageFilter
import numpy as np
import matplotlib.pyplot as plt


# Compensating for bluish tint by adjusting both Red and Blue channels
def compensate_blue(image):
    imager, imageg, imageb = image.split()
    
    # Get min and max values for each channel
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()
    
    # Convert to arrays
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    
    x, y = image.size
    
    # Normalize the pixel values to the range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = (imageR[i][j] - minR) / (maxR - minR)
            imageG[i][j] = (imageG[i][j] - minG) / (maxG - minG)
            imageB[i][j] = (imageB[i][j] - minB) / (maxB - minB)
    
    # Get mean values for each channel
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Adjust Green and Red channels based on the Blue channel
    for i in range(y):
        for j in range(x):
            imageR[i][j] = int((imageR[i][j] + (meanB - meanR) * (1 - imageR[i][j]) * imageB[i][j]) * maxR)
            imageG[i][j] = int((imageG[i][j] + (meanB - meanG) * (1 - imageG[i][j]) * imageB[i][j]) * maxG)
    
    # Scale Blue channel back to the original range
    for i in range(0, y):
        for j in range(0, x):
            imageB[i][j] = int(imageB[i][j] * maxB)
    
    # Create the compensated image array
    blue_compensated = np.zeros((y, x, 3), dtype="uint8")
    blue_compensated[:, :, 0] = imageR  # Red channel
    blue_compensated[:, :, 1] = imageG  # Green channel
    blue_compensated[:, :, 2] = imageB  # Blue channel
    
    # Plot the compensated image
    # plt.figure(figsize=(20, 20))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image (Bluish)")
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.title("Green and Red Compensated Image")
    # plt.imshow(blue_compensated)
    # plt.show()

    # Return the image as a PIL Image object
    return Image.fromarray(blue_compensated)



