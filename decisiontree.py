import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Kiểm tra và tải tài nguyên nltk nếu cần thiết
try:
    nltk.data.find('tokenizers/punkt')
    print("Tài nguyên 'punkt' đã được tải xuống.")
except LookupError:
    print("Tài nguyên 'punkt' chưa được tải xuống. Vui lòng thử tải lại.")
    nltk.download('punkt')

# Đọc dữ liệu từ file CSV
file_path = "mail_data.csv"  # Đảm bảo đường dẫn đúng đến file của bạn
df = pd.read_csv(file_path)

# Gán cột EmailText và Label
message_X = df["Message"].values  # Cột chứa nội dung thư
labels_Y = df["Category"].replace({"ham": 1, "spam": 0}).values  # Gán cột Label

# Khởi tạo stemmer và hàm xử lý tin nhắn
lstem = LancasterStemmer()

def preprocess_messages(messages):
    processed_messages = []
    for message in messages:
        # Lọc các ký tự không phải chữ cái
        message = ''.join(filter(lambda char: (char.isalpha() or char == " "), message))
        # Tokenize và chuyển về từ gốc
        words = word_tokenize(message)
        processed_message = ' '.join([lstem.stem(word) for word in words])
        processed_messages.append(processed_message)
    return processed_messages

# Tiền xử lý tin nhắn và tạo vector hóa
processed_messages = preprocess_messages(message_X)
tfvec = TfidfVectorizer(stop_words='english')
X = tfvec.fit_transform(processed_messages).toarray()  # Chuyển đổi tin nhắn thành các vector
y = labels_Y

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Khởi tạo và huấn luyện mô hình Decision Tree
dt_classifier = DecisionTreeClassifier(max_depth=5)
dt_classifier.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = dt_classifier.predict(X_test)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")

# Lưu mô hình và vectorizer
pickle.dump(dt_classifier, open("decision_tree_model.pkl", "wb"))
pickle.dump(tfvec, open("feature_extraction_decision_tree.pkl", "wb"))