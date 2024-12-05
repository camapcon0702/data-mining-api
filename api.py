from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from googleapiclient.discovery import build
import mysql.connector
from datetime import datetime
import schedule
import time
import threading
import requests
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import seaborn as sns
import pandas as pd

from crawl import update_database, engine

app = Flask(__name__)

# Cấu hình kết nối với MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:matkhau1@localhost/dongtaypromotion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Tắt việc theo dõi các thay đổi đối với database

# Khởi tạo SQLAlchemy với Flask app
db = SQLAlchemy(app)

# Tạo model ví dụ cho bảng PromotionData
class PromotionData(db.Model):
    __tablename__ = 'promotiondata'
    
    # id = db.Column(db.Integer, primary_key=True)
    Title = db.Column(db.String(255), primary_key=True)
    Published_date = db.Column(db.Date)
    Views = db.Column(db.Integer)
    Likes = db.Column(db.Integer)
    Comments = db.Column(db.Integer)

    def __init__(self, title, publish_date, views, likes, comments):
        self.Title = title
        self.Published_date = publish_date
        self.Views = views
        self.Likes = likes
        self.Comments = comments
        
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)
        
def get_data_frame():
    query = "SELECT * FROM promotiondata"
    df = pd.read_sql(query, con=engine)
    
    df['Published_date'] = pd.to_datetime(df['Published_date']).dt.date
    
    df['Month'] = pd.to_datetime(df['Published_date']).dt.strftime('%b')
    return df

# url = "http://127.0.0.1:8080/"

# def get_crawl_data():
#     response = requests.get(url)
#     if response.status_code == 200:
#         print(response.json())  # In ra phản hồi từ API
#     else:
#         print(f"Failed to call API. Status code: {response.status_code}")

@app.route('/')
def crawl_data():
    print("Bắt đầu chương trình cập nhật dữ liệu...")
    update_database()  # Chạy lần đầu ngay lập tức
    schedule.every(5).minutes.do(update_database)

    # Chạy `schedule` trong một luồng riêng nếu chưa khởi chạy
    if not any(thread.name == "ScheduleThread" for thread in threading.enumerate()):
        schedule_thread = threading.Thread(target=run_schedule, name="ScheduleThread", daemon=True)
        schedule_thread.start()

    return f"[{datetime.now()}] Đã khởi động chương trình cập nhật dữ liệu định kỳ."

def classification():
    data = get_data_frame()
    label_encoder = preprocessing.LabelEncoder()
    data['Month'] = label_encoder.fit_transform(data['Month'])
    
    print(data)
    
    data.dropna()

    conditions = [
        (data['Views'] < 1000000),
        (data['Views'] >= 1000000) & (data['Views'] < 5000000),
        (data['Views'] >= 5000000)
    ]
    choices = ['Low', 'Medium', 'High']
    data['Target'] = pd.cut(data['Views'], bins=[0, 1000000, 5000000, float('inf')], labels=choices)

    # Chọn các đặc trưng và mục tiêu
    X = data[['Views', 'Likes', 'Comments', 'Month']]
    y = data['Target']

    # Tiền xử lý và chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Áp dụng RandomForestClassifier để phân loại
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá mô hình
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # In kết quả phân loại
    data['Predicted_Target'] = model.predict(X_scaled)
    
    print(data)
    
    return data

def video_count():
    df = get_data_frame()
    
    keywords = ['2 Ngày 1 Đêm', 'Nhanh Như Chớp', 'Ngôn Ngữ Diệu Kỳ', 'Thực Khách Vui Vẻ', 'Quýt Làm Cam Chịu', 'Find My Fan', 'Ngạc Nhiên Chưa', 'Ký Ức Vui Vẻ', 'Tâm sự cùng tổng đài', 'A! Đúng Rồi']


    keyword_counts = {keyword: df['Title'].str.contains(keyword, case=False, na=False).sum() for keyword in keywords}

    # Chuyển đổi kết quả thành DataFrame để dễ dàng vẽ biểu đồ
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Video Count'])

    # Lọc các từ khóa có số lượng video > 0 để biểu đồ rõ ràng hơn
    keyword_df = keyword_df[keyword_df['Video Count'] > 0]
    
    return keyword_df

def keywords_count():
    
    df = get_data_frame()
    
    keywords = ['giải trí', 'Việt Nam', 'OUR SONG', 'ĂN', 'thực tế', 'CƯỜI', 'hài', 'SCHOOL', 'FAN', 'TÂM SỰ', 'CA SĨ', 'VÀNG', 'Vui Vẻ', 'CHUYỆN TÌNH', 'HÁT', 'ĐÀN ÔNG', 'ĐÙA', 'BẤT NGỜ', 'ĐỘ HÓT', 'HỒ SƠ', 'DẤU VẾT', 'HỘI NGỘ']
    
    keyword_counts = {keyword: df['Title'].str.contains(keyword, case=False, na=False).sum() for keyword in keywords}

    # Chuyển đổi kết quả thành DataFrame để dễ dàng vẽ biểu đồ
    keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Video Count'])

    # Lọc các từ khóa có số lượng video > 0 để biểu đồ rõ ràng hơn
    keyword_df = keyword_df[keyword_df['Video Count'] > 0]
    
    return keyword_df



@app.route('/get_keyword', methods=['GET'])
def get_keyword():
    df = keywords_count()
    
    result = df.to_dict(orient='records')
    return jsonify(result)


@app.route('/view_promotion')
def view_promotion():
    promotions = PromotionData.query.all()
    return '<br>'.join([f"{promo.Title}, {promo.Published_date}, {promo.Views}, {promo.Likes}, {promo.Comments}" for promo in promotions])

@app.route('/get_promotion_data', methods=['GET'])
def get_promotion_data():
    try:
        # Lấy dữ liệu từ bảng promotiondata
        df = get_data_frame()
        
        # Chuyển đổi DataFrame thành JSON để trả về
        result = df.to_dict(orient='records')
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route('/get_views_by_month', methods=['GET'])
def get_views_by_month():
    try:
        df = get_data_frame()
        view_by_month = df.groupby('Month')['Views'].sum().sort_index()
        
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        view_by_month_df = view_by_month.reset_index()

        view_by_month_df['Month'] = pd.Categorical(view_by_month_df['Month'], categories=month_order, ordered=True)
        view_by_month_df = view_by_month_df.sort_values('Month')
        
        result = view_by_month_df.to_dict(orient='records')
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    

@app.route('/get_clustering', methods=['GET'])
def get_cluster():
    try:
        data = get_data_frame()
    
        data['Like_to_View_Ratio'] = data['Likes'] / data['Views']

        # Loại bỏ các dòng có giá trị NaN (ví dụ như video có Views = 0)
        data = data.dropna(subset=['Like_to_View_Ratio'])

        # Chuẩn hóa đặc trưng tỷ lệ Like/Views
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[['Like_to_View_Ratio']])

        # Áp dụng KMeans phân cụm
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scaled_features)
        
        # data = data.reset_index()
        
        result = data.to_dict(orient='records')
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
    
def classify_video(row):
    if row['Views'] < 1000000:
        return 'Low'
    elif row['Views'] < 5000000:
        return 'Medium'
    else:
        return 'High'

# Áp dụng phân loại cho mỗi dòng dữ liệu trong DataFrame

@app.route('/get_classification', methods=['GET'])
def get_classification():
    # Convert the DataFrame to a dictionary
    data = get_data_frame()
    
    data['Target'] = data.apply(classify_video, axis=1)
    result = data[['Title', 'Views', 'Published_date', 'Likes', 'Comments', 'Target']].to_dict(orient='records')
    return jsonify(result)
        
@app.route('/get_video_count', methods=['GET'])
def get_video_count():
    df = video_count()
    
    # data = df.reset_index()
    
    result = df.to_dict(orient='records')
    return jsonify(result)
    
if __name__ == "__main__":
    app.run(debug=True, port=8080)
    
