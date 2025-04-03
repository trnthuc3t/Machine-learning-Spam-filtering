import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
raw_mail_data = pd.read_csv("mail_data.csv")
raw_mail_data.head()
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

raw_mail_data.isnull().sum()
df = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
df.isnull().sum()
df.shape
# Supervised -> target class
# Unsupervised -> clustering problem

# Label encoding
df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})
df.head()

# loc function
df.loc[df['Category'] == 'spam', 'Category',] = 0
df.loc[df['Category'] == 'ham', 'Category',] = 1
df.head()
X = df['Message']
Y = df['Category']
X
Y
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape
y_train.shape
X_test.shape
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", binary=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
df.info()
y_train = y_train.astype("int")
y_test = y_test.astype("int")
X_train
print(X_train_features)
print(X_test_features)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_features, y_train)
prediction_train_data = model.predict(X_train_features)
accuracy_train_data = accuracy_score(y_train, prediction_train_data)
print("Accuarcy on train data: ", accuracy_train_data)
prediction_test_data = model.predict(X_test_features)
accuracy_test_data = accuracy_score(y_test, prediction_test_data)
print("Accuarcy on test data: ", accuracy_test_data)
input_user_mail = [
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]

input_data_features = feature_extraction.transform(input_user_mail)

prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("This is a ham mail")
else:
    print("This is a spam mail")
    import pickle

    pickle.dump(model, open("logistic_regressio.pkl", "wb"))
    pickle.dump(feature_extraction, open("feature_extraction_logistic_regressio.pkl", "wb"))