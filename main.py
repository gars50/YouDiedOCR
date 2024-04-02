from yt_download import download_yt
from process_vid import process


#target_url = input('Paste the link to the video to check')
target_url = 'https://www.youtube.com/watch?v=aUANPtlh7pw'
download_yt(target_url)
file_to_process = 'toprocess.webm'

#process(file_to_process)