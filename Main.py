from flask import Flask, request
import sys      
sys.path.append('src/')

from ImageDetector import ImageDetector

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

'''@app.route("/image")
def hello():
    value = request.args['color']
    imgDet = ImageDetector("Resources/images/MyPic.jpg")
    data = imgDet.getImageMat(color= value)
    return data, 200'''

if __name__=="__main__":
    app.run()
    #imgDet = ImageDetector("Resources/images/MyPic.jpg")
    #imgDet.BlurImage()
