import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import nltk
nltk.download('punkt_tab')

# Kiểm tra và tải tài nguyên nltk nếu cần thiết
try:
    nltk.data.find('tokenizers/punkt')
    print("Tài nguyên 'punkt' đã được tải xuống.")
except LookupError:
    nltk.download('punkt')

# Đọc dữ liệu từ file CSV
file_path = "mail_data.csv"  # Cập nhật đường dẫn file nếu cần
df = pd.read_csv(file_path)

# Xử lý dữ liệu
df = df[['Category', 'Message']]  # Giữ lại 2 cột quan trọng
df['Category'] = df['Category'].map({'ham': 1, 'spam': 0})  # Chuyển 'ham' thành 1, 'spam' thành 0

# Kiểm tra nếu có dữ liệu bị null và thay thế bằng chuỗi rỗng
df = df.fillna('')

# Tiền xử lý văn bản
lstem = LancasterStemmer()

def preprocess_messages(messages):
    processed_messages = []
    for message in messages:
        # Loại bỏ ký tự không phải chữ cái
        message = ''.join(filter(lambda char: char.isalpha() or char == " ", message))
        # Tokenize và chuyển từ về dạng gốc
        words = word_tokenize(message)
        processed_message = ' '.join([lstem.stem(word) for word in words])
        processed_messages.append(processed_message)
    return processed_messages

# Tiền xử lý tin nhắn
processed_messages = preprocess_messages(df['Message'])

# Vector hóa dữ liệu văn bản
# tfvec = TfidfVectorizer(stop_words='english')
tfvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9)

X = tfvec.fit_transform(processed_messages).toarray()
y = df['Category'].values  # Nhãn đã được mã hóa

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Khởi tạo và huấn luyện mô hình Random Forest**
# rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
rf_classifier = RandomForestClassifier(
    n_estimators=200,   # Số lượng cây trong rừng
    max_depth=20,       # Độ sâu tối đa của cây
    min_samples_split=10, # Số mẫu tối thiểu để chia một node
    max_features='sqrt', # Sử dụng số đặc trưng tối đa
    random_state=42,
    class_weight="balanced"
)

rf_classifier.fit(X_train, y_train)

# **Dự đoán và đánh giá mô hình**
y_pred = rf_classifier.predict(X_test)

# **Tính toán các chỉ số đánh giá**
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Độ chính xác (Accuracy): {accuracy*100:.2f}%")
print(f"Độ chính xác (Precision): {precision*100:.2f}%")
print(f"Độ nhạy (Recall): {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")

print("\nBáo cáo phân loại:\n", classification_report(y_test, y_pred, target_names=['Spam', 'Ham']))

# **Lưu mô hình và vectorizer**
pickle.dump(rf_classifier, open("random_forest_model.pkl", "wb"))
pickle.dump(tfvec, open("feature_extraction_random_forest.pkl", "wb"))