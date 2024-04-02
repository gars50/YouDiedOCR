from yt_download import download_yt
from process_vid import process
from tkinter import *
import os
import json

with open('config.json', 'r') as f:
    config = json.load(f)
debug_mode = (str.lower(config['debug_mode']) == "true") or (str.lower(config['debug_mode']) == "enabled")

if debug_mode:
    #Ensure folders are in place for images
    if not os.path.exists("debug_images"):
        os.makedirs("debug_images")



target_url = 'https://www.youtube.com/watch?v=aUANPtlh7pw'
file_downloaded = download_yt(target_url)
file_downloaded = 'toprocess.webm'

process(file_downloaded, 'ring')