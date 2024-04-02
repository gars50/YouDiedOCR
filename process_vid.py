import pytesseract
import cv2
import numpy as np
import json
import os

#https://stackoverflow.com/questions/66993242/make-faster-videocapture-opencv
#https://github.com/Jan-9C/deathcounter_ocr/blob/main/deathcounter.py

#Fetch configs
with open('config.json', 'r') as f:
    config = json.load(f)

with open(config["crop_file"], 'r') as f:
    crop = json.load(f)

with open(config["mask_file"]) as f:
    mask_file = json.load(f)

tesseract_directory_path = config['tesseract_directory']
debug_mode = config['debug_mode']
refresh_time = float(config['refresh_time'])
refresh_time_death = int(config['refresh_time_death'])
ocr_string = config["ocr_string"]
language = config["language"]
levenshtein_d = int(config["levensthein_d"])

# Set tesseract path to exe
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_directory_path, "tesseract.exe")

# Function to generate levensthein distance between two Strings
# in other words it returns how much the strings differ
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]

def check_frame_for_death(image):
    is_death = False

    # Image crop coodinates
    x=int(crop["x"])
    y=int(crop["y"])
    width=int(crop["width"])
    height=int(crop["height"])

    # Crop the image
    image = image[y:y+height, x:x+width]

    # Debug Info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/cropped.png", image)

    # Color the image
    image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2HSV_FULL)

    # Debug info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/cropped_colored.png", image)

    firstMaskFetched = False;

    ## TODO: Optimize fetching of values so that it isnt executed every time
    for element in mask_file:
        lower = np.array(element["lower"])
        pixelvalue = np.array(element["pixel"])
        upper = np.array(element["upper"])
        mask_lower = cv2.inRange(image, lower, pixelvalue)
        mask_upper = cv2.inRange(image, pixelvalue, upper)
        if firstMaskFetched:
            mask = mask + mask_lower + mask_upper
        else:
            mask = mask_lower + mask_upper
            firstMaskFetched = True;

    output_img = image.copy()
    output_img[np.where(mask==0)] = 0

    image = output_img

    # Debug Info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/mask.png", image)

    # Turn image grayscale
    image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2GRAY)

    # Debug Info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/image_grayscale.png", image)

    # TODO: Improve image processing? Probably not perfect?
    # Black and White processing
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #Apply dilation and erosion to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
    image = cv2.GaussianBlur(image, (5,5), 0)

    # Read text from image
    imgtext = pytesseract.image_to_string(image, lang=language, config='--psm 11 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_pageseg_mode=1 -c tessedit_min_word_length=2')

    # get levenshtein distance of complete cropped image
    ldistance = levenshtein(imgtext, ocr_string)

    # Get the shape of the image
    blackheight, blackwidth = np.shape(image)

    # Create a black image with the same shape as the input image
    black_image = np.zeros((blackheight, blackwidth, 3), dtype=np.uint8)
    black_image = cv2.cvtColor(np.array(black_image),cv2.COLOR_BGR2GRAY)

    # The following process generates 2 new images
    # One where the right halft is filled with black pixels and one where the left half is filled with black pixels
    # Overall this can help the OCR Algorithm as the simplifed images have less noise

    # Copy the image so that the right half can be filled with black pixels
    imageBlackR = image.copy()

    # Fill the left half of the image with black pixels
    imageBlackR[:, :width//2] = black_image[:, :width//2]

    # Debug Info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/imageBlackR.png", imageBlackR)

    # Copy the image so that the left half can be filled with black pixels
    imageBlackL = image.copy()

    # Fill the right half of the image with black pixels
    imageBlackL[:, width//2:] = black_image[:, width//2:]

    # Debug Info
    if(debug_mode == "enabled"):
        cv2.imwrite("debugImages/images/imageBlackL.png", imageBlackL)

    # Pass both processed simplified images to the OCR Algorithm
    righthalftext = pytesseract.image_to_string(imageBlackR, lang=language, config='--psm 11 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_pageseg_mode=1 -c tessedit_min_word_length=2')
    lefthalftext = pytesseract.image_to_string(imageBlackL, lang=language, config='--psm 11 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -c tessedit_pageseg_mode=1 -c tessedit_min_word_length=2')

    # Get the levensthein distance of the recognized text
    right_ldistance = levenshtein(righthalftext, ocr_string)
    left_ldistance = levenshtein(lefthalftext, ocr_string)

    # Choose the smallest levensthein distance as the true levenshtein distance / Choose the closest match
    ldistance = min(ldistance, right_ldistance, left_ldistance)

    # Debug Info
    if(debug_mode == "enabled"):
        print("Detected: " + lefthalftext + "|" + imgtext + "|" + righthalftext)
        print("ldistance: " + str(ldistance))

    # Check for acceptable levenshtein distance
    if ldistance >= levenshtein_d:
        # Debug Info
        if(debug_mode == "enabled"):
            print("No valid text found")
            cv2.imwrite("debugImages/images/unsuccessfull.png", image)

    elif ldistance < levenshtein_d:
        # Debug Info
        if(debug_mode == "enabled"):
            print("Valid Text found: " + lefthalftext + "|" + imgtext + "|" + righthalftext)

        is_death = True

        # Debug Info
        if(debug_mode == "enabled"):
             cv2.imwrite("debugImages/images/successfull.png", image)

    return is_death

def add_death(timestamp):
    print(f'Death at {timestamp}')

def process(target_video_file):
    video = cv2.VideoCapture(target_video_file)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    next_needed_frame_no = 0
    current_frame_no = 0

    while video.isOpened():
        is_death = False
        ret = video.grab()
        if not ret:
            #Video has ended
            break
            
        if (next_needed_frame_no == current_frame_no):
            status, frame = video.retrieve()
            is_death = check_frame_for_death(frame)
            if is_death:
                add_death(str(video.get(cv2.CAP_PROP_POS_MSEC)))
                next_needed_frame_no = current_frame_no + refresh_time_death*video_fps
            else:
                next_needed_frame_no = current_frame_no + refresh_time*video_fps
                print(f'No death found at {str(video.get(cv2.CAP_PROP_POS_MSEC))}')
        else:
            current_frame_no+=1
        
        if cv2.waitKey(10) & 0XFF == ord('q'):
            break

    video.release()

process('toprocess.webm')