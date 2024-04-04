import cv2
import numpy as np
import time
import threading
import yt_dlp

class ProcessThread(threading.Thread):
    def __init__(self, url, game, debug_option, milliseconds_skipped, seconds_skipped_after_death):
        self.progress_status = ""
        self.phase = "Initializing"
        self.youtube_url = url
        self.game = game
        self.debug = debug_option
        self.video_location = ""
        self.milliseconds_skipped = milliseconds_skipped
        self.seconds_skipped_after_death = seconds_skipped_after_death
        self.death_timestamps = ""
        super().__init__()
    
    def run(self):
        #self.download_yt()
        #self.process_vid()
        for f in range(100):
            time.sleep(1)
            self.progress_status = f'Phase : {self.phase}\nPercent done : {f}'
    
    def download_yt(self):
        ydl_opts = {'format': 'bestvideo',
                    'quiet': 'true',
                    'outtmpl': 'toprocess.%(ext)s'}
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info_dict = ydl.extract_info(self.youtube_url, download=False)
        video_title = info_dict.get('title', None)
        print(f'Downloading video : {video_title}')
        ydl.download(self.youtube_url)
    
    def add_death(self, timestamp):
        new_death = f'Death at {timestamp}\n'
        self.death_timestamps = f'{self.death_timestamps}{new_death}'
    
    def process(self):
        video = cv2.VideoCapture(self.video_location)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        video_total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        next_needed_frame_no = 0
        current_frame_no = 0
        start_time = time.time()

        while video.isOpened():
            is_death = False
            ret = video.grab()
            if not ret:
                #Video has ended
                break
                
            if (next_needed_frame_no == current_frame_no):
                status, frame = video.retrieve()
                is_death = check_frame_for_death(frame, self.game)
                if is_death:
                    self.add_death(str(video.get(cv2.CAP_PROP_POS_MSEC)))
                    next_needed_frame_no = current_frame_no + self.refresh_time_death*video_fps
                else:
                    next_needed_frame_no = current_frame_no + self.refresh_time*video_fps
            else:
                current_frame_no+=1
            
            if cv2.waitKey(10) & 0XFF == ord('q'):
                break
            
            if (current_frame_no % 30 == 0):
                fps = current_frame_no/(time.time() - start_time)
                if fps == 0:
                    fps = 1
                print(f'Processing at {round(fps, 2)} frames per seconds. \nTime remaining : {round(((video_total_frames - current_frame_no) / fps), 2)} seconds')
        video.release()


#https://stackoverflow.com/questions/66993242/make-faster-videocapture-opencv
#https://github.com/Jan-9C/deathcounter_ocr/blob/main/deathcounter.py

lower_red1 = (0, 67, 80)
upper_red1 = (5, 255, 255)
lower_red2 = (170, 67, 80)
upper_red2 = (180, 255, 255)

def check_frame_for_death(image, game_type, debug_mode):
    is_death = False

    if debug_mode:
        start = time.time()
    # __red color mask__
    # __DARK SOULS 1, 2, 3__ __DEMONS SOULS REMAKE__ __SEKIRO SHADOW DIE TWICE__
    # img_ori = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_BGRA2RGB)
    img_ori = np.array(image)
    original_image = img_ori
    img_ori = cv2.resize(img_ori, dsize=(960, 540), interpolation=cv2.INTER_AREA)
    img_hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(img_hsv, lower_red1, upper_red1)
    img_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    img_mask3 = img_mask + img_mask2
    img_result = cv2.bitwise_and(img_ori, img_ori, mask=img_mask3)
    contours, _ = cv2.findContours(img_mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug_mode:
        cv2.imwrite("debug_images/img_ori.png", img_result)
        cv2.imwrite("debug_images/img_hsv.png", img_result)
        cv2.imwrite("debug_images/img_mask.png", img_result)
        cv2.imwrite("debug_images/img_mask2.png", img_result)
        cv2.imwrite("debug_images/img_mask3.png", img_result)
        cv2.imwrite("debug_images/img_result.png", img_result)
        
    height, width, channels = img_ori.shape
    w_rate = width / 1920
    h_rate = height / 1080
    # Square box
    fX = []
    D = []
    contour_array = []
    blue_box = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Create a vertical rectangle
        if game_type == "souls" \
                and 70 * h_rate < h < 140 * h_rate \
                and 10 * w_rate < w < 110 * w_rate:
            contour_array.append([x, y, w, h])
        elif game_type == "sekiro" and 200 < h < 350 and 50 < w < 350:
            contour_array.append([x, y, w, h])
        elif game_type == "ring" and 50 * h_rate < h < 80 * h_rate and 10 * w_rate < w < 80 * w_rate:
            contour_array.append([x, y, w, h])
        # __ sort countours
    for i in range(len(contour_array) - 1):
        index = i
        for j in range(i + 1, len(contour_array)):
            # print(centers[j])
            if contour_array[index][0] > contour_array[j][0]:
                index = j
        contour_array[i], contour_array[index] = contour_array[index], contour_array[i]
    # __ y position average
    contour_y_sum = 0
    avg = 0
    if game_type == "souls" or game_type == "ring":
        delete_array = []
        for i in range(len(contour_array)):
            contour_y_sum = contour_y_sum + contour_array[i][1]
        try:
            avg = int(contour_y_sum / len(contour_array))
        except ZeroDivisionError:
            avg = 0
        # print("avg : ", avg)
        for i in range(len(contour_array)):
            # __ contour_array[i][1] : contours y position
            sub = avg - contour_array[i][1]
            if sub < 0: # sub = int(np.sqrt(sub * sub))
                sub *= -1
            if sub > 80 * h_rate:
                delete_array.append(i)
        for i in reversed(delete_array):
            del contour_array[i]
        delete_array.clear()

    contour_y_sum = 0
    for i in range(len(contour_array)):
        contour_y_sum = contour_y_sum + contour_array[i][1]
    try:
        avg = int(contour_y_sum / len(contour_array))
    except ZeroDivisionError:
        avg = 0
    
    if debug_mode:
        print("len contour_array : ", len(contour_array))
        print("contour_array : ", contour_array)
    for i in range(len(contour_array)):
        # print("i : ", i)
        # __ (average - contour_y position) < 50 ? bluebox.append : next
        sub = avg - contour_array[i][1]
        if sub < 0: # sub = int(np.sqrt(sub * sub))
            sub *= -1
        if sub < 50 * h_rate:
            blue_box.append(contour_array[i])
            if debug_mode:
                cv2.rectangle(
                    img_result,
                    (contour_array[i][0], contour_array[i][1]),
                    (contour_array[i][0] + contour_array[i][2], contour_array[i][1] + contour_array[i][3]),
                    (255, 0, 0), 2)
        # ### Find the distance [D] between contours & the vertical ratio of the contours must be constant.
    if len(blue_box) >= 2:  # ##if len(centers) >= 2: if len(contour_array) >= 2:
        for idx in range(len(blue_box) - 1):
            dx = blue_box[idx][0] - blue_box[idx + 1][0]
            dy = blue_box[idx][1] - blue_box[idx + 1][1]
            D.append(int(np.sqrt(dx * dx + dy * dy)))
    
    if debug_mode:
        print("D : ", D)

    for i in range(len(D)):
        if game_type == "souls" and D[i] < 160 * w_rate: # __ souls
            fX.append(D[i])
        elif game_type == "sekiro" and 160 < D[i] < 165: # __ sekiro
            fX.append(D[i])
        elif game_type == "ring" and D[i] < 50 * w_rate * 2:
            fX.append(D[i])

    if debug_mode:
        print(fX)

    if game_type == "souls" and 5 < len(fX) < 9:
        is_death = True
    elif game_type == "sekiro" and len(fX) == 1:
        is_death = True
    elif game_type == "ring" and 5 < len(fX) < 9:
        is_death = True
    contour_array.clear()
    blue_box.clear()
    D.clear()
    fX.clear()
    
    if debug_mode:
        print(f'Processing : {round(1 / (time.time() - start), 2)} frames per seconds')

    return is_death