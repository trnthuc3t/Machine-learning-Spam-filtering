import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Đọc dữ liệu từ file CSV
raw_mail_data = pd.read_csv("mail_data.csv")
raw_mail_data.head()

# Kiểm tra dữ liệu thiếu
raw_mail_data.isnull().sum()

# Thay thế dữ liệu thiếu bằng chuỗi rỗng
df = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
df.isnull().sum()

# Chuyển đổi nhãn Category thành 0 và 1
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})
df.head()

# Tách dữ liệu thành X (tin nhắn) và Y (nhãn)
X = df['Message']
Y = df['Category']

# Chia tập dữ liệu thành training và test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Sử dụng TfidfVectorizer để chuyển văn bản thành vector đặc trưng
from sklearn.feature_extraction.text import TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", binary=True)

# Chuyển đổi dữ liệu huấn luyện và kiểm tra thành vector
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Kiểm tra loại dữ liệu
df.info()

# Đảm bảo nhãn là kiểu int
y_train = y_train.astype("int")
y_test = y_test.astype("int")

# Import và huấn luyện mô hình SVM
from sklearn.svm import SVC
model = SVC(kernel='linear')

# Huấn luyện mô hình
model.fit(X_train_features, y_train)

# Dự đoán và tính toán độ chính xác trên tập huấn luyện
prediction_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train, prediction_train_data)
print("Accuracy on train data: ", accuracy_train_data)

# Dự đoán và tính toán độ chính xác trên tập kiểm tra
prediction_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, prediction_test_data)
print("Accuracy on test data: ", accuracy_test_data)

# Dự đoán cho một email của người dùng
input_user_mail = [
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]

# Chuyển đổi dữ liệu đầu vào thành vector
input_data_features = feature_extraction.transform(input_user_mail)

# Dự đoán email là spam hay ham
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("This is a ham mail")
else:
    print("This is a spam mail")

# Lưu mô hình SVM và vectorizer đã huấn luyện
import pickle
pickle.dump(model, open("svm_model.pkl", "wb"))
pickle.dump(feature_extraction, open("feature_extraction_svm.pkl", "wb"))