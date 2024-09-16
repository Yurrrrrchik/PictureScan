import sqlite3
import logging
import cv2
import numpy as np
import base64
import threading
import json
import os
from flask import Flask, render_template, request, redirect, session, make_response
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask_sqlalchemy import SQLAlchemy
from SCAN4 import scan
from tkinter import Tk, filedialog
from multiprocessing import Process

root = Tk()
root.withdraw()

was_scanning = False

### loggers ###

logger = logging.getLogger('PictureScannerLogger')
logger.setLevel(logging.DEBUG)

# Создаем обработчик для записи логов в файл
file_handler = logging.FileHandler('PictureScannerLog.log')
file_handler.setLevel(logging.DEBUG)

# Создаем обработчик для вывода логов в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Создаем форматтер для логов
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Добавляем обработчики к логгеру
logger.addHandler(file_handler)
logger.addHandler(console_handler)

### flask, db ###

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///PictureScannerDB.db'  # Пример URI для SQLite
db = SQLAlchemy(app)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    total_scans = db.Column(db.Integer)

    def get_id(self):
        return str(self.id)


class Scans(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_image = db.Column(db.LargeBinary)
    original_image_name = db.Column(db.Text)
    denoised_image = db.Column(db.LargeBinary)
    scanned_text = db.Column(db.Text)


class UsersScans(db.Model):
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), primary_key=True)


with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            logger.info('Failed attempt to enter with blank field(s)')
            return render_template('Entering.html', error_message="Login and password fields shouldn't be empty")

        if 'create' in request.form:
            existing_user = User.query.filter_by(username=username).first()

            if existing_user:
                return render_template('Entering.html', error_message='User with this username already exists')

            new_user = User(username=username, password=password, total_scans=0)
            db.session.add(new_user)
            db.session.commit()
            logger.info(f'User {username} is successfully registered in the system')

            return redirect('/main_page')

        elif 'enter' in request.form:
            user = User.query.filter_by(username=username, password=password).first()

            if user:
                login_user(user)
                logger.info(f'User {username} entered the system')
                return redirect('/main_page')
            else:
                logger.info('Failed attempt to enter')
                return render_template('Entering.html', error_message='Invalid login and/or password')

    return render_template('Entering.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    logger.info(f'User {current_user.username} logged out')
    logout_user()
    return redirect('/')


@app.route('/main_page', methods=['GET', 'POST'])
@login_required
def main_page():
    return render_template('MainPage.html')


@app.route('/scanning_form', methods=['GET', 'POST'])
@login_required
def scanning_form():
    input_image_name = ''
    input_image_base64 = ''
    enhanced_image_base64 = ''
    extracted_text = ''
    if request.method == 'POST':
        if 'scan' in request.form:
            input_image_name = request.files['file']
            language = request.form['language']
            if input_image_name.filename == "":
                return render_template('ScanningForm.html')
            input_image = cv2.imdecode(np.frombuffer(input_image_name.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image, extracted_text = scan(input_image, lang=language)
            _, img_encoded = cv2.imencode('.png', processed_image)
            _, input_image = cv2.imencode('.png', input_image)
            # input_image_bytes = input_image.tobytes()
            # img_encoded_bytes = img_encoded.tobytes()
            enhanced_image_base64 = base64.b64encode(img_encoded).decode('utf-8')
            input_image_base64 = base64.b64encode(input_image).decode('utf-8')

            new_scan = Scans(original_image=input_image, original_image_name=input_image_name.filename,
                            denoised_image=img_encoded, scanned_text=extracted_text)
            db.session.add(new_scan)
            db.session.commit()
            new_user_scan = UsersScans(user_id=current_user.id, scan_id=new_scan.id)
            db.session.add(new_user_scan)
            db.session.commit()
            cur_user = User.query.get(current_user.id)
            cur_user.total_scans += 1
            db.session.commit()

            logger.info(f'User {current_user.username} made a scan')

        if 'save' in request.form:
            last_scan = Scans.query.order_by(Scans.id.desc()).first()
            image_process = Process(target=save_image, args=(last_scan.original_image_name, last_scan.scanned_text,
                                                            current_user.username))
            image_process.start()
    return render_template('ScanningForm.html', original_image=input_image_base64,
                           processed_image=enhanced_image_base64,
                           extracted_text=extracted_text)



@app.route('/home', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.filter_by(username=current_user.username).first()
    if user:
        user_scans = Scans.query.join(UsersScans).filter_by(user_id=user.id).all()
        for scan in user_scans:
            # scan.original_image = scan.original_image.to
            scan.original_image = base64.b64encode(scan.original_image).decode('utf-8')
            scan.denoised_image = base64.b64encode(scan.denoised_image).decode('utf-8')
    if request.method == 'POST':
        new_username = request.form['username']
        new_password = request.form['password']

        if new_username:
            user_with_new_username = User.query.filter_by(username=new_username).first()
            if user_with_new_username:
                return render_template('Profile.html', error_message='Invalid login and/or password')
            old_username = user.username
            user.username = new_username
        if new_password:
            user.password = new_password
        if new_password or new_username:
            db.session.commit()
        logger.info(f'{old_username} have changed login to {new_username}')
        logger.info(f'{user.username} have changed their password')
    return render_template('Profile.html', user_scans=user_scans, user=current_user)


def save_image(img, extracted_text, user):
    image_path = filedialog.asksaveasfilename(initialdir="/", title="Choose a folder to save denoised picture",
                                              defaultextension=".json")
    if image_path:
        data = {img: extracted_text}
        with open(image_path, "w") as json_file:
            json.dump(data, json_file)

    if os.path.exists(image_path):
        logger.info(f'User {user} saved .json file to disk')


if __name__ == '__main__':
    app.run(debug=False)