
import os
from warnings import filterwarnings
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image


UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            model=load_model('VGG16_model.h5')
            filename = secure_filename(file.filename)
            file.save(str(app.config['UPLOAD_FOLDER'])+ filename)
            img_file=image.load_img(str(app.config['UPLOAD_FOLDER'])+ filename,target_size=(224,224))
            x= image.img_to_array(img_file)
            x=np.expand_dims(x,axis=0)
            img_data=preprocess_input(x)
            result=model.predict(img_data)
            
            if result[0][0]>0.5:
                prediction = "The patient under examination is COVID-19 NEGATIVE. You are advised to continue adhering to Covid-19 guidelines."
                return render_template('index.html', prediction= prediction)
             


              
            else:
             
             prediction = "The patient under examination is COVID-19 POSITIVE.....you are advised to quarantine within this medical area and adhere to all medical proceedings as per command by the medical team."



            return render_template('index.html', prediction= prediction)
    return render_template('index.html')

if __name__ == '__main__':

    app.run(debug=True)
