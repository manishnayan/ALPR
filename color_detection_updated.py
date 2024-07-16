import joblib
import numpy as np
import mahotas
import cv2
import time

clf = joblib.load("/home/manish/ANPR/base_modules/color_model.pkl")

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic


def fd_histogram(image, mask=None):
    bins = 8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist,hist)
    return hist.flatten()

def plate_color(img):
    img = img[12:-12,25:-25]
    global_feature = []
    fixed_size  = tuple((100,100))
    img = cv2.resize(img, fixed_size)
    fv_hu_moments = fd_hu_moments(img)
    fv_haralick   = fd_haralick(img)
    fv_histogram  = fd_histogram(img)
    global_feature = np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    prediction = int(prediction)
    if prediction==0:
        return 'white'
    elif prediction==1:
        return 'yellow'
    
    
def check(image):
    return plate_color(image)
    
#if __name__=="__main__":
#    image = cv2.imread("./plate.jpg")

#    p = time.time()
#    a= plate_color(image)
#    print(a)
