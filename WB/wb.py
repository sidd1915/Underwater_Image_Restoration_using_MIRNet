import numpy as np
from PIL import Image
from .compensateRB import compensate_RB
from .compensateblue import compensate_blue

# Function to detect dominant color channel
def detect_dominant_tint(image):
    imager, imageg, imageb = image.split()
    meanR = np.mean(np.array(imager))
    meanG = np.mean(np.array(imageg))
    meanB = np.mean(np.array(imageb))
    
    # Determine dominant color tint based on channel means
    if meanG > meanB and meanG > meanR:
        return 'greenish'
    elif meanB > meanG and meanB > meanR:
        return 'bluish'
    else:
        return 'neutral'
    
# Function to compensate based on tint
def compensate_image(image):
    tint = detect_dominant_tint(image)
    
    if tint == 'greenish':
        #print("Applying greenish compensation...")
        return compensate_RB(image, 0)  # Compensating red and blue using green channel
    elif tint == 'bluish':
        #print("Applying bluish compensation...")
        return compensate_blue(image)  # Compensating blue using red and green channels
    else:
        #print("No dominant tint detected, skipping compensation.")
        return image  # Return the original image if neutral


# Gray World Algorithm for white balancing
def gray_world(image):
    imager, imageg, imageb = image.split()
    imagegray = image.convert('L')
    
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)
    
    x, y = image.size
    
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)

    
    
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = int(imageR[i][j] * meanGray / meanR)
            imageG[i][j] = int(imageG[i][j] * meanGray / meanG)
            imageB[i][j] = int(imageB[i][j] * meanGray / meanB)
    
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR
    whitebalancedIm[:, :, 1] = imageG
    whitebalancedIm[:, :, 2] = imageB
    
    # plt.figure(figsize=(20, 20))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(image)
    # plt.subplot(1, 2, 2)
    # plt.title("White Balanced Image")
    # plt.imshow(whitebalancedIm)
    # plt.show()

    return Image.fromarray(whitebalancedIm)


 # Example usage
# image_path = r"C:\Users\Siddhi Patil\Downloads\15603.png"
# image = Image.open(image_path).convert('RGB')

# # First, perform color compensation based on tint
# compensated_image = compensate_image(image)

# # Then, apply white balancing to the compensated image
# white_balanced_image = gray_world(compensated_image)

# white_balanced_image.save(r"C:\Users\Siddhi Patil\Downloads\15630wb.png")