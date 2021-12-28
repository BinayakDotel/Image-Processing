from flask import Flask, request, Response
from matplotlib.pyplot import cool
from numpy.lib.type_check import imag
from werkzeug.datastructures import auth_property
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import os, time
import numpy as np

import sys

from werkzeug.wrappers import response      
sys.path.append('src/')
from ImageOperation import ImageOperation
from ImageClassifier import ImageClassifier

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]= 'sqlite:///Datas.sqlite3'
app.config["UPLOAD_IMAGE"] = "D:/MyProject/static/images/"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]= False

db= SQLAlchemy(app)

class Img(db.Model):
    id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True)
    img = db.Column(db.Text, unique=True, nullable= False)
    name = db.Column(db.Text, nullable= False)
    mimetype = db.Column(db.Text, nullable= False)

    def __init__(self, img, name, mimetype):
        self.img = img
        self.name = name
        self.mimetype = mimetype

@app.route('/conversion_types')
def conversionTypes():
    img_operation = ImageOperation("static\images\MyPic.JPG")
    data = img_operation.getConversionTypes()
    return data, 200

@app.route("/image")
def hello():
    value = request.args['color']
    img_operation = ImageOperation("static\images\MyPic.JPG")
    data = img_operation.getImageMat(color= value)
    return data, 200

#Here, we upload the image to the database
@app.route("/upload_image", methods=["POST"])
def uploadImage():
    pic = request.files['pic']
    if not pic:
        return 'No pic uploaded!', 400
    
    #We get image in string format
    imgData = pic.read()

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400

    #Create object of the image having id, name, img, and mimetype
    dbImg = Img(img= imgData, name=filename, mimetype=mimetype)

    try:
        #Adding the dbImg to the image
        db.session.add(dbImg)

        #And, finally committing the data to the database (UPLOADING)
        db.session.commit()
    except:
        print("IMAGE ALREADY EXISTS!")

    return f"<h1>{filename}, Uploaded Successfully </h1>", 200

# This takes id as parameter and returns the image 
@app.route('/<string:name>')
def get_img(name):
    value = request.args['color']
    img = Img.query.filter_by(name=name).first()
    if not img:
        return 'Img Not Found!', 404

    im = ImageOperation("ga")
    computedImg = im.getImagePixel(img.img, value)

    return Response(computedImg, mimetype=img.mimetype)

if __name__=="__main__":
    #app.run(host='192.168.1.81') #Running flask server to an specific host (i.e Default gateway of my PC)
    #db.create_all()
    #app.run()

    img_operation= ImageOperation("TestImages/Sneaker.jpg")
    image= img_operation.readImage()
    img_operation.showImage(image, frameName="Original Image")

    image= img_operation.Convert2Gray(image)

    #Since our model accepts only image with shape (1, 28, 28), resize the image first
    image= img_operation.ResizeImage(image, height= 28)

    #Cropping the image to make it center
    image= img_operation.cropTo(image)

    img_operation.showImage(image, frameName="Cropped")
    print(image.shape)
    
    normalized_img = (image.astype(np.float32) / 127.0) - 1
    print(normalized_img)


    #This process is done to create an stack of images
    conv_pool= []
    stack_pool= []

    rows= 3
    cols= 3
    for i in range(1, (rows*cols)+1):
        conv_img= img_operation.CNNFromScratch(image, filter_size=3)
        conv_pool.append(conv_img)

        if i%cols==0:
            stack_pool.append(conv_pool)
            conv_pool=[]
        
    img_operation.showImage(img_operation.CreateStack(stack_pool), frameName="Stack")
    
    #Creating an image array of shape as per our model 
    data_for_model = np.ndarray(shape=(1, 28, 28), dtype=np.float32)

    #Copying the image to initial image
    data_for_model[0]= normalized_img

    img_classifier= ImageClassifier()

    #Predict accepts list of image datas, i.e. data_for_model
    img_classifier.predict(data_for_model)

    img_operation.WaitForClosing()

