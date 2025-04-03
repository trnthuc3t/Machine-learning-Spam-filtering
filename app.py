from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
import pickle
import os
import json
import base64
import webbrowser

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
app = Flask(__name__)
app.secret_key = 'super-secret-key-12345'

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_FILE = 'credentials.json'

models = {
    'logistic_regression': pickle.load(open('logistic_regression.pkl', 'rb')),
    'random_forest': pickle.load(open('random_forest_model.pkl', 'rb')),
    'naive_bayes': pickle.load(open('naive_bayes_model.pkl', 'rb')),
    'svm': pickle.load(open('svm_model.pkl', 'rb')),
    'decision_tree': pickle.load(open('decision_tree_model.pkl', 'rb'))
}

feature_extraction_files = {
    'logistic_regression': 'feature_extraction_logistic_regressio.pkl',
    'random_forest': 'feature_extraction_random_forest.pkl',
    'naive_bayes': 'feature_extraction_naive_bayes.pkl',
    'svm': 'feature_extraction_svm.pkl',
    'decision_tree': 'feature_extraction_decision_tree.pkl'
}


def get_gmail_service():
    creds = None
    try:
        if os.path.exists('token.json'):
            with open('token.json', 'r') as token_file:
                creds_data = json.load(token_file)
                creds = Credentials(**creds_data)
        if not creds or not creds.valid:
            if 'credentials' not in session or not all(k in session['credentials'] for k in
                                                       ['token', 'refresh_token', 'token_uri', 'client_id',
                                                        'client_secret']):
                return None
            creds_data = session['credentials']
            creds = Credentials(
                token=creds_data['token'],
                refresh_token=creds_data['refresh_token'],
                token_uri=creds_data['token_uri'],
                client_id=creds_data['client_id'],
                client_secret=creds_data['client_secret'],
                scopes=creds_data['scopes']
            )
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                session['credentials']['token'] = creds.token
                with open('token.json', 'w') as token_file:
                    json.dump(session['credentials'], token_file)
        return build('gmail', 'v1', credentials=creds)
    except Exception:
        return None


def predict_mail(input_text):
    input_user_mail = [input_text]
    results = {}
    for model_name in models.keys():
        try:
            feature_extraction = pickle.load(open(feature_extraction_files[model_name], 'rb'))
            input_data_features = feature_extraction.transform(input_user_mail)
            prediction = models[model_name].predict(input_data_features)[0]
            results[model_name] = prediction
        except Exception:
            pass  # Bỏ qua lỗi mà không ghi log
    return results


def get_email_content(service, msg_id):
    try:
        msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        headers = msg['payload']['headers']
        subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
        sender = next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown Sender')

        content = ''
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
                    break
        else:
            content = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8', errors='replace')

        return {'id': msg_id, 'subject': subject, 'sender': sender, 'content': content, 'snippet': msg['snippet']}
    except Exception:
        return None


def get_emails(service):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
        messages = results.get('messages', [])
        inbox = []
        spam = []
        for msg in messages:
            email_data = get_email_content(service, msg['id'])
            if email_data:
                predictions = predict_mail(email_data['content'])
                # 0 là thư rác, 1 là thư thường
                spam_models = [model for model, pred in predictions.items() if pred == 0]
                email_data['spam_models'] = spam_models
                if spam_models:
                    spam.append(email_data)
                else:
                    inbox.append(email_data)
        return inbox, spam
    except Exception:
        return [], []


@app.route('/')
def index():
    service = get_gmail_service()
    if service is None:
        return redirect(url_for('authorize'))
    inbox, spam = get_emails(service)
    return render_template('index.html', inbox=inbox, spam=spam)


@app.route('/login')
def login():
    return redirect(url_for('authorize'))


@app.route('/authorize')
def authorize():
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        flow.redirect_uri = url_for('oauth2callback', _external=True)
        authorization_url, state = flow.authorization_url(access_type='offline', prompt='consent')
        session['state'] = state
        webbrowser.open(authorization_url)
        return redirect(authorization_url)
    except Exception:
        return "Error in authorization", 500


@app.route('/oauth2callback')
def oauth2callback():
    try:
        state = session.get('state')
        if not state:
            raise Exception("State not found in session")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES, state=state)
        flow.redirect_uri = url_for('oauth2callback', _external=True)
        authorization_response = request.url
        flow.fetch_token(authorization_response=authorization_response)
        creds = flow.credentials
        session['credentials'] = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        with open('token.json', 'w') as token_file:
            json.dump(session['credentials'], token_file)
        return redirect(url_for('index'))
    except Exception:
        return "Error during authentication", 500


@app.route('/update_emails')
def update_emails():
    service = get_gmail_service()
    if service is None:
        return jsonify({'error': 'Not authenticated', 'redirect': url_for('authorize')})
    inbox, spam = get_emails(service)
    return jsonify({'inbox': inbox, 'spam': spam})


if __name__ == '__main__':
    app.run(debug=True, port=5000)