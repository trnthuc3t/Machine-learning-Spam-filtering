import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer

# Kiểm tra tài nguyên 'punkt' của nltk và tải xuống nếu chưa có
try:
    nltk.data.find('tokenizers/punkt')
    print("Tài nguyên 'punkt' đã được tải xuống.")
except LookupError:
    print("Tài nguyên 'punkt' chưa được tải xuống. Vui lòng thử tải lại.")
    nltk.download('punkt')

# Đọc dữ liệu từ file CSV
file_path = "mail_data.csv"
df = pd.read_csv(file_path)

# Gán cột EmailText và Label
message_X = df.iloc[:, 1]  # Cột EmailText
labels_Y = df.iloc[:, 0]   # Cột Label

# Khởi tạo Stemmer
lstem = LancasterStemmer()

# Hàm xử lý tin nhắn
def mess(messages):
    message_x = []
    for me_x in messages:
        me_x = ''.join(filter(lambda mes: (mes.isalpha() or mes == " "), me_x))  # Lọc dữ liệu
        words = word_tokenize(me_x)  # Chia nhỏ các từ
        message_x += [' '.join([lstem.stem(word) for word in words])]  # Gom nhóm từ gốc
    return message_x

# Tiền xử lý các tin nhắn
message_x = mess(message_X)

# Sử dụng TfidfVectorizer để vector hóa các tin nhắn
tfvec = TfidfVectorizer(stop_words='english')
x_new = tfvec.fit_transform(message_x).toarray()

# Chuyển đổi nhãn ham và spam thành giá trị 0 và 1
y_new = np.array(labels_Y.replace(to_replace=['ham', 'spam'], value=[1, 0]).astype(int))

# Tách dữ liệu thành training và test set
x_train, x_test, y_train, y_test = ttsplit(x_new, y_new, test_size=0.2, random_state=1)

# Sử dụng mô hình MultinomialNB (Naive Bayes)
classifier = MultinomialNB()

# Huấn luyện mô hình
classifier.fit(x_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = accuracy_score(y_test, classifier.predict(x_test))
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Tính toán và hiển thị Confusion Matrix
cmat = confusion_matrix(y_test, classifier.predict(x_test))
print('Confusion Matrix is: \n', cmat)

# Lưu mô hình và bộ trích xuất đặc trưng vào file
pickle.dump(classifier, open("naive_bayes_model.pkl", "wb"))
pickle.dump(tfvec, open("feature_extraction_naive_bayes.pkl", "wb"))