import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pydensecrf.densecrf as dcrf

# File I/O
INPUT_PATH = 'D:/Datasets/penguinguy/%03d%03d.png'
OUTPUT_PATH = "output/penguinguy_crf_8"

# Arch Camera Configurlation
TOTAL_CAMERA = 10
TOTAL_IMAGE_PER_CAMERA = 40
THRESHOLD_RATIO = 0.05 # Difference betweem image [0 - 1]

# Rotate
USE_ROTATE_TO_IMAGE = True
ROTATE_DIRECTION = cv2.ROTATE_90_CLOCKWISE

# Remove noise when differnece 2 image
USE_REMOVE_PEPPER_NOISE = True
OPENNING_KERNEL_SIZE = (5,5)

THRESHOLD_STRENG = 20
THRESHOLD_USE_TRIANGLE = False

USE_CLOSING_FOREGROUND = True
CLOSING_KERNEL = (30,30)

MP_POOL_SIZE = 10

# CRF configure
CRF_TOTAL_LABEL = 2

#Maximize varience
FOREGROUND_CLUSTER = 4
BACKGROUND_CLUSTER = 4

# BACKGROUND
USE_BLUR_EDGE = False
BLUR_KERNEL = (11,11)
BLUR_ERODE_KERNEL = (5,5)

USE_MEAN_BACKGROUND = True
NEW_BACKGROUND_COLOR = (0,0,0) # range 0-255 / active when not use mean background

USE_DENOISE_FOREGROUND_MASK = True
FOREGROUND_OPENNING_KERNEL_SIZE = (7,7)

CRF_PAIRWISE_GAUSSIAN_SXY = 3
CRF_PAIRWISE_GAUSSIAN_COMPACT = 3
CRF_PAIRWISE_BILATERAL_SXY = 80
CRF_PAIRWISE_BILATERAL_SRGB = 13
CRF_PAIRWISE_BILATERAL_COMPACT = 10
CRF_ITERATION = 20
    

def get_diff_mask(camerea_id,current_shot):
    previous_shot = (current_shot - 1) % TOTAL_IMAGE_PER_CAMERA
    # read image
    image_prev_uint = cv2.imread(INPUT_PATH % (camerea_id,previous_shot)) 
    image_current_uint = cv2.imread(INPUT_PATH % (camerea_id,current_shot)) 
    # convert to RGB
    image_prev_uint = cv2.cvtColor(image_prev_uint,cv2.COLOR_BGR2RGB)
    image_current_uint = cv2.cvtColor(image_current_uint,cv2.COLOR_BGR2RGB)
    # rotate
    if USE_ROTATE_TO_IMAGE:
        image_prev_uint = cv2.rotate(image_prev_uint, ROTATE_DIRECTION)
        image_current_uint = cv2.rotate(image_current_uint, ROTATE_DIRECTION)
    # convert to [0-1]
    image_prev =  image_prev_uint / 255.0
    image_current = image_current_uint / 255.0
    # difference mask between 2 images
    diff_mask = np.linalg.norm(image_current - image_prev, axis=-1)
    diff_mask = (diff_mask > THRESHOLD_RATIO) * 1.0
    #remove noise from sensor (hope this not ruin image)
    if USE_REMOVE_PEPPER_NOISE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,OPENNING_KERNEL_SIZE)
        denoised_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    else:
        denoised_mask = diff_mask
    return denoised_mask

def background_model(CAMERA_NUMBER):
    print("Intializing Camera:%02d" % (CAMERA_NUMBER, ))
    foreground_prob = get_diff_mask(CAMERA_NUMBER,0)
    for i in range(1,40):
        foreground_prob = cv2.bitwise_or(foreground_prob,get_diff_mask(CAMERA_NUMBER,i))
    if USE_CLOSING_FOREGROUND:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,CLOSING_KERNEL)
        mask_closed = cv2.morphologyEx(foreground_prob, cv2.MORPH_CLOSE, kernel)
    else:
        mask_closed = foreground_prob
    # find boundary (rectangle) of object 
    mask_y, mask_x = np.nonzero(mask_closed)
    min_x = np.min(mask_x)
    max_x = np.max(mask_x)
    min_y = np.min(mask_y)
    max_y = np.max(mask_y)

    #flood fill to remove hole
    image_flooded = (mask_closed.copy() * 255.0).astype(np.uint8)
    image_height, image_width = image_flooded.shape[:2]
    flood_mask = np.zeros((image_height+2,image_width+2),dtype=np.uint8)
    has_flooded = False
    # top bar
    if min_y != 0:
        for i in range(image_flooded.shape[1]):
            if image_flooded[0,i] != 255:
                has_flooded = True
                cv2.floodFill(image_flooded, flood_mask, (i,0), 255)
    # left bar
    if min_x != 0:
        for i in range(image_flooded.shape[0]):
            if image_flooded[i,0] != 255:
                has_flooded = True
                cv2.floodFill(image_flooded, flood_mask, (0,i), 255)

    # right bar
    most_right = image_flooded.shape[1] -1
    if max_x != most_right:
        for i in range(image_flooded.shape[0]):
            if image_flooded[i,most_right] != 255:
                has_flooded = True
                cv2.floodFill(image_flooded, flood_mask, (most_right,i), 255)

    # bottom bar 
    most_bottom = image_flooded.shape[0] -1
    if max_y != most_bottom:
        for i in range(image_flooded.shape[1]):
            if image_flooded[most_bottom,i] != 255:
                has_flooded = True
                cv2.floodFill(image_flooded, flood_mask, (i,most_bottom), 255)

    # we get background from floodfill
    if has_flooded:
        background_mask = flood_mask[1:-1,1:-1]
    else:
        background_mask = 1 - mask_closed
    is_background = background_mask == 1

    # backgroud mm model 
    default_image = cv2.imread(INPUT_PATH % (CAMERA_NUMBER,0)) 
    if USE_ROTATE_TO_IMAGE:
        default_image = cv2.rotate(default_image, ROTATE_DIRECTION)
    default_image = default_image / 255.0

    background_mm_model = cv2.ml.EM_create()
    background_mm_model.setClustersNumber(BACKGROUND_CLUSTER)
    background_mm_model.trainEM(default_image[is_background])
    cv2.imwrite("background_mask_{:02}.png".format(CAMERA_NUMBER),background_mask * 255)
    fs = cv2.FileStorage("background_mm_model_{:02}.model".format(CAMERA_NUMBER),cv2.FILE_STORAGE_WRITE)
    background_mm_model.write(fs)
    fs.release()

def find_foreground(compress_parameter):
    CAMERA_NUMBER, IMAGE_NUMBER = compress_parameter
    background_mask = cv2.imread("background_mask_{:02}.png".format(CAMERA_NUMBER),cv2.IMREAD_UNCHANGED)  
    is_background = background_mask == 255
    background_mm_model = cv2.ml.EM_create()
    fs = cv2.FileStorage("background_mm_model_{:02}.model".format(CAMERA_NUMBER),cv2.FILE_STORAGE_READ)
    background_mm_model.read(fs.getNode(''))
    fs.release()
    print("working on Cam:%02d, Shot:%02d" % (CAMERA_NUMBER, IMAGE_NUMBER))
    image_current_uint = cv2.imread(INPUT_PATH % (CAMERA_NUMBER,IMAGE_NUMBER)) 
    if USE_ROTATE_TO_IMAGE:
        image_current_uint = cv2.rotate(image_current_uint, ROTATE_DIRECTION) 
    HEIGHT, WIDTH, CHANNEL = image_current_uint.shape
    #Threshold for foreground
    image_gray = cv2.cvtColor(image_current_uint,cv2.COLOR_BGR2GRAY)
    if THRESHOLD_USE_TRIANGLE:
        ret2,object_threshold = cv2.threshold(image_gray,200,255,cv2.THRESH_TRIANGLE)
    else:
        ret2,object_threshold = cv2.threshold(image_gray,THRESHOLD_STRENG,255,cv2.THRESH_BINARY)        

    if USE_DENOISE_FOREGROUND_MASK:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,FOREGROUND_OPENNING_KERNEL_SIZE)
        object_threshold_opened = cv2.morphologyEx(object_threshold, cv2.MORPH_OPEN, kernel)
    else:
        object_threshold_opened = object_threshold
    
    is_foreground = object_threshold_opened > 0

    # normalize image to float32 [0,1]
    image_current_float32 = image_current_uint / 255.0

    # unknown mask
    unknown_mask = np.ones((HEIGHT, WIDTH))
    unknown_mask[is_background] = 0
    unknown_mask[is_foreground] = 0
    is_unknown = unknown_mask > 0

    #Forground mm model
    foreground_mm_model = cv2.ml.EM_create()
    foreground_mm_model.setClustersNumber(FOREGROUND_CLUSTER)
    foreground_mm_model.trainEM(image_current_float32[is_foreground])
    foreground_weights = foreground_mm_model.getWeights()

    # prediction foreground
    _, foreground_predicteds = foreground_mm_model.predict(image_current_float32[is_unknown])
    foreground_probability = foreground_predicteds.dot(foreground_weights.T)

    # prediction background
    background_weights = background_mm_model.getWeights()
    _, background_predicteds = background_mm_model.predict(image_current_float32[is_unknown])
    background_probability = background_predicteds.dot(background_weights.T)

    # Foreground probability map
    foreground_probability_map = np.zeros((HEIGHT, WIDTH))
    foreground_probability_map[is_unknown] = foreground_probability[:,0]
    foreground_probability_map[is_foreground] = 1.0
    foreground_probability_map[is_background] = 0.0

    # Background probability map
    background_probability_map = np.zeros((HEIGHT, WIDTH))
    background_probability_map[is_unknown] =  background_probability[:,0]
    background_probability_map[is_foreground] = 0.0
    background_probability_map[is_background] = 1.0

    # DENSE CRF
    denseCRF = dcrf.DenseCRF2D(WIDTH, HEIGHT, CRF_TOTAL_LABEL)
    unary = np.dstack((foreground_probability_map,background_probability_map))
    unary = -np.log(unary) # denseCRF require negative log probability
    unary = unary.astype(np.float32) #require float32
    unary = unary.transpose(2, 0, 1).reshape((CRF_TOTAL_LABEL,-1)) # unary need to be flat.
    unary = np.ascontiguousarray(unary) #avoid cython problem :X
    denseCRF.setUnaryEnergy(unary)

    image_current_rgb = cv2.cvtColor(image_current_uint,cv2.COLOR_BGR2RGB)
    denseCRF.addPairwiseGaussian(
        sxy=CRF_PAIRWISE_GAUSSIAN_SXY,
        compat=CRF_PAIRWISE_GAUSSIAN_COMPACT
    )
        
    denseCRF.addPairwiseBilateral(
        sxy=CRF_PAIRWISE_BILATERAL_SXY,
        srgb=CRF_PAIRWISE_BILATERAL_SRGB,
        rgbim=image_current_rgb,
        compat=CRF_PAIRWISE_BILATERAL_COMPACT
    )

    segmented_probability = denseCRF.inference(CRF_ITERATION)
    segmented_mask = np.argmax(segmented_probability, axis=0).reshape((HEIGHT,WIDTH))
    segmented_foreground = segmented_mask == 0
    segmented_background = segmented_mask == 1

    # Find background color
    if USE_MEAN_BACKGROUND:
        background_color = np.mean(image_current_uint[segmented_background],axis=0)
    else:
        background_color = NEW_BACKGROUND_COLOR
        
    output_image = image_current_uint.copy()
    output_image[segmented_background] = background_color
    if USE_BLUR_EDGE:
        output_image = cv2.blur(output_image, BLUR_KERNEL)
        new_mask = 1 - segmented_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BLUR_ERODE_KERNEL)
        eroded_mask = cv2.erode(new_mask.astype(np.uint8),kernel)
        remain_foreground = eroded_mask == 1
        output_image[remain_foreground] = image_current_uint[remain_foreground]

    #create output image
    cv2.imwrite("%s/cam%03d_%05d.png"%(OUTPUT_PATH,CAMERA_NUMBER,IMAGE_NUMBER), output_image)    

def cleanup_model():
    files = os.listdir('.')
    mask_files = [ f for f in files if f.startswith('background_mask_')]
    model_files = [f for f in files if f.startswith('background_mm_model_')]
    remove_files = mask_files + model_files
    for f in remove_files:
        os.remove(f)

def cleanup_output():
    files = os.listdir(OUTPUT_PATH)
    for f in files:
        os.remove(os.path.join(OUTPUT_PATH,f))

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    print("finding background mm model")
    params = list(range(TOTAL_CAMERA))
    pool = Pool(MP_POOL_SIZE)  
    pool.map(background_model, params)  
    print("finding foreground")
    params = []
    for i in range(TOTAL_CAMERA):
        for j in range(TOTAL_IMAGE_PER_CAMERA):
            params.append((i, j))
    pool = Pool(MP_POOL_SIZE)  
    pool.map(find_foreground, params)  
    cleanup_model()
    # cleanup_output()

