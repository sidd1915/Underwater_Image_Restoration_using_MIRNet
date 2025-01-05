import numpy as np
import cv2
# Taken from: https://github.com/huabo-zhu/underwater-advancement
from .sceneRadianceCLAHE import RecoverCLAHE
#from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')

def apply_clahe(img):
        sceneRadiance = RecoverCLAHE(img)
        # cv2.imwrite('OutputImages/' + prefix + '_CLAHE.jpg', sceneRadiance)
        return sceneRadiance
