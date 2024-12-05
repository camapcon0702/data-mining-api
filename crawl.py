from sqlalchemy import create_engine
from googleapiclient.discovery import build
import mysql.connector
from datetime import datetime
import schedule
import time
import seaborn as sns
import pandas as pd

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'matkhau1',
    'database': 'youtube_data'
}
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Cài đặt YouTube API
API_KEY = 'AIzaSyCvGdKhweS8R0V2z1xVSRqUrU-lx5ooscA'  # Thay bằng API Key của bạn

CHANNEL_ID = 'UCFMEYTv6N64hIL9FlQ_hxBw'
playlist_id = 'UUFMEYTv6N64hIL9FlQ_hxBw'

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_ids(youtube, playlist_id):
    
    request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId = playlist_id,
                maxResults = 50)
    response = request.execute()
    
    video_ids = []
    
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
        
    next_page_token = response.get('nextPageToken')
    more_pages = True
    
    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                        part='contentDetails',
                        playlistId = playlist_id,
                        maxResults = 50,
                        pageToken = next_page_token)
            response = request.execute()
    
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
        
    return video_ids


def get_video_details(youtube, video_ids):
    all_video_stats = []
    
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids[i:i+50]))
        response = request.execute()
        
        for video in response['items']:
            video_stats = dict(
                Title = video['snippet']['title'],
                Published_date = video['snippet']['publishedAt'],
                Views = video['statistics'].get('viewCount', 0),  # Sử dụng .get() với giá trị mặc định là 0
                Likes = video['statistics'].get('likeCount', 0),   # Giá trị mặc định nếu không có likeCount
                # Dislikes = video['statistics'].get('dislikeCount', 0),
                Comments = video['statistics'].get('commentCount', 0)
            )
            all_video_stats.append(video_stats)
    
    return all_video_stats




username = 'root'
password = 'matkhau1'
host = 'localhost'
database = 'dongtaypromotion'

# Tạo kết nối tới MySQL
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}/{database}')


def update_database():
    print(f"[{datetime.now()}] Bắt đầu cào dữ liệu...")
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    video_ids = get_video_ids(youtube, playlist_id)
    video_details = get_video_details(youtube, video_ids)

    video_data = pd.DataFrame(video_details)
    video_data['Published_date'] = pd.to_datetime(video_data['Published_date']).dt.date
    video_data['Views'] = pd.to_numeric(video_data['Views'])
    video_data['Likes'] = pd.to_numeric(video_data['Likes'])
    video_data['Comments'] = pd.to_numeric(video_data['Comments'])

    # Ghi dữ liệu vào MySQL
    try:
        video_data.to_sql('promotiondata', con=engine, if_exists='replace', index=False)
        print(f"[{datetime.now()}] Dữ liệu đã được cập nhật thành công!")
    except Exception as e:
        print(f"Lỗi khi ghi dữ liệu vào database: {e}")
        
        
    # Lên lịch chạy hàm mỗi 5 phút
    # schedule.every(5).minutes.do(update_database)