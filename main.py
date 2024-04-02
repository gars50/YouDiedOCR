from yt_download import download_yt
from process_vid import process
import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)
debug_mode = config['debug_mode']
if debug_mode:
    #Ensure folders are in place
    if not os.path.exists("debug_images"):
        os.makedirs("debug_images")

#target_url = input('Paste the link to the video to check')
target_url = 'https://www.youtube.com/watch?v=aUANPtlh7pw'
download_yt(target_url)
file_to_process = 'toprocess.webm'

#process(file_to_process)