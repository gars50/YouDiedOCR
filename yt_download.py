import yt_dlp

def download_yt(target_url):
    ydl_opts = {'format': 'bestvideo',
                'quiet': 'true',
                'outtmpl': 'toprocess.%(ext)s'}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(target_url, download=False)
    video_title = info_dict.get('title', None)
    print(f'Downloading video : {video_title}')
    ydl.download(target_url)