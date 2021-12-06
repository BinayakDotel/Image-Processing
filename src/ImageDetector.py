import cv2 as cv
import math
import base64
import numpy as np
from matplotlib import pyplot as plot

class ImageDetector:
    def __init__(self, path):
        self.path = path
        print(f"Searching image at {self.path}.")
    
    def getImageMat(self, color):
        img = cv.imread(self.path)
        width, height, channel= img.shape
        width/=2
        height/=2
        cv.circle(img, (int(height),int(width)), 20, (255,0,0), 2)
        if color=='BGR':
            print(f"WE GOT::{color}")
            data = base64.b64encode(cv.imencode('.jpg', img)[1]).decode()
        
        elif color == 'RGB':
            print(f"WE GOT::{color}")
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            data = base64.b64encode(cv.imencode('.jpg', rgb_img)[1]).decode()
        
        elif color == 'GRAY':
            print(f"WE GOT::{color}")
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            data = base64.b64encode(cv.imencode('.jpg', gray_image)[1]).decode()
        
        else:
            color='BGR'
            print(f"WE GOT::{color}")
            data = base64.b64encode(cv.imencode('.jpg', img)[1]).decode()
        dict = {
            color: data
        }
        return dict

    #Function to convert decimal number into binary number
    def DecimalToBinary(self, num):
        d=""
        if num >= 1:
            self.DecimalToBinary(num // 2)

        data = num % 2
        
    def BinaryToDecimal(self, num):
        a=  int(f"0b{num}", base=0)
        return a

    def convertToBW(self):
        img = cv.imread(self.path)
        color = img.copy()
        cv.imshow("Original",img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        k = 150

        width, height = img.shape
        bw= np.zeros((width, height))
        negative = img.copy()
        blur = img.copy()
        logImg= img.copy()
        powLowImg= img.copy()
        powHighImg= img.copy()

        for x in range(0, width):
            for y in range(0, height):
                if img[x,y] < k:
                    bw[x,y] = 0
                else:
                    bw[x,y] = 255
                
            #img is converted to Digital Negative image
                scaling = img[x,y]/255
                dn = 1-scaling
                rescaling = int(dn * 255)
                negative[x,y] = rescaling

            #Logarithmic Transformation -> S=c*log(1+r)
                logImg[x,y] = 20 * math.log(1+img[x,y])

            #Power law Transformation -> S=c * pow(r, gama)
                powLowImg[x,y] = 20 * math.pow(img[x,y], 0.4)
                powHighImg[x,y] = 2 * math.pow(img[x,y], 2.0)

        b0 = (img >> 0) & 1
        b1 = (img >> 1) & 1
        b2 = (img >> 2) & 1
        b3 = (img >> 3) & 1
        b4 = (img >> 4) & 1
        b5 = (img >> 5) & 1
        b6 = (img >> 6) & 1

        mm= (color>>0) & 1
        print(f"mm::{mm}")
        
        for x in range(0, width):
            for y in range(0, height):
                b0[x,y] = self.BinaryToDecimal(f"000000{b0[x,y]}")
                b1[x,y] = self.BinaryToDecimal(f"00000{b1[x,y]}0")
                b2[x,y] = self.BinaryToDecimal(f"0000{b2[x,y]}00")
                b3[x,y] = self.BinaryToDecimal(f"000{b3[x,y]}000")
                b4[x,y] = self.BinaryToDecimal(f"00{b4[x,y]}0000")
                b5[x,y] = self.BinaryToDecimal(f"0{b5[x,y]}00000")
                b6[x,y] = self.BinaryToDecimal(f"{b6[x,y]}000000")

        print(f"IMG::{img}")
        print(f"b0::{b0}")
        print(f"b1::{b1}")
        print(f"b2::{b2}")
        print(f"b3::{b3}")
        print(f"b4::{b4}")
        print(f"b5::{b5}")
        print(f"b6::{b6}")

        cv.imshow("IMG", img)
        cv.imshow("LSB", b0)
        cv.imshow("1st", b1)
        cv.imshow("2nd", b2)
        cv.imshow("3rd", b3)
        cv.imshow("4th", b4)
        cv.imshow("5th", b5)
        cv.imshow("MSB", b6)

        cv.imshow("Negative", negative)
        cv.imshow("Black", bw)
        cv.imshow("Gray", img)
        cv.imshow("LOG IMG", logImg)
        cv.imshow("POW LOW IMG", powLowImg)
        cv.imshow("POW HIGH IMG", powHighImg)
      

        # This section is responsible to perform constrast/Histogram Stretching
        #   (New Image) S= [((Smax - Smin)/(Rmax - Rmin))*(R - Rmin) + Smin] (R->Original Image)
        StretchHistImg = img.copy()
        m = (255-0)/(248-1)
        StretchHistImg[:,:] = m * (img[:,:]-1)
        cv.imshow("Contrast stretch", StretchHistImg)

        plot.hist(img.ravel(), 255, [0, 255])
        plot.hist(StretchHistImg.ravel(), 255, [0, 255])
        '''plot.hist(logImg.ravel(), 255, [0, 255])
        plot.hist(powLowImg.ravel(), 255, [0, 255])
        plot.hist(powHighImg.ravel(), 255, [0, 255])'''

        plot.show()

        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

    def BlurImage(self):
        img = cv.imread(self.path)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = gray_img.copy()
        blur01 = gray_img.copy()
        sober_img = gray_img.copy()
        width, height = gray_img.shape

        #   _________________________________
        #   | x-1, y-1 | x, y-1 | x+1, y-1  |
        #   | x-1, y   | x, y   | x+1, y    |
        #   | x-1, y+1 | x, y+1 | x+1, y+1  |
        #   |__________|________|___________| 

        top_left = 0
        top = 0
        top_right = 0
        left = 0
        right = 0
        bottom_left = 0
        bottom_right = 0
        bottom = 0
        center = 0

        for x in range(0, width-1):
            for y in range(0, height-1):
                if x == 0 and y == 0:
                    top         = 0
                    top_left    = 0 
                    top_right   = 0
                    left        = 0
                    bottom_left = 0
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                    bottom          = gray_img[x,y+1]
                    bottom_right    = gray_img[x+1,y+1]

                elif x == 0 and y == height:
                    top_left        = 0 
                    left            = 0
                    bottom_left     = 0
                    bottom          = 0
                    bottom_right    = 0
                    top         = gray_img[x,y-1]
                    top_right   = gray_img[x+1,y-1]
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                
                elif x == width and y == height:
                    top_right        = 0 
                    right            = 0
                    bottom_left     = 0
                    bottom          = 0
                    bottom_right    = 0
                    top         = gray_img[x,y-1]
                    top_left    = gray_img[x-1,y-1] 
                    left        = gray_img[x-1,y]
                    center      = gray_img[x,y]

                elif y == 0:
                    top         = 0
                    top_left    = 0 
                    top_right   = 0
                    left        = gray_img[x-1,y]
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                    bottom_left     = gray_img[x-1,y+1]
                    bottom          = gray_img[x,y+1]
                    bottom_right    = gray_img[x+1,y+1]
                
                elif x == 0:
                    left        = 0
                    top_left    = 0
                    bottom_left = 0
                    top         = gray_img[x,y-1]
                    top_right   = gray_img[x+1,y-1]
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                    bottom          = 0
                    bottom_right    = 0

                elif x == width:
                    top_right          = 0
                    bottom_right    = 0 
                    right = 0
                    top         = gray_img[x,y-1]
                    top_left    = gray_img[x-1,y-1] 
                    left        = gray_img[x-1,y]
                    center      = gray_img[x,y]
                    bottom_left     = gray_img[x-1,y+1]
                    bottom          = gray_img[x,y+1]

                elif y == height:
                    bottom_left     = 0
                    bottom          = 0
                    bottom_right    = 0
                    top         = gray_img[x,y-1]
                    top_left    = gray_img[x-1,y-1] 
                    top_right   = gray_img[x+1,y-1]
                    left        = gray_img[x-1,y]
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                
                else:
                    top         = gray_img[x,y-1]
                    top_left    = gray_img[x-1,y-1] 
                    top_right   = gray_img[x+1,y-1]
                    left        = gray_img[x-1,y]
                    center      = gray_img[x,y]
                    right       = gray_img[x+1,y]
                    bottom_left = gray_img[x-1,y+1]
                    bottom      = gray_img[x,y+1]
                    bottom_right= gray_img[x+1,y+1]

                '''
                    Performing Gaussian blur
                    kernel= | 1  2  1 |
                            | 2  4  2 |
                            | 1  2  1 |
                '''
                blur[x,y] = (top_left * 1 + top *4 + top_right*1 + left*4 +
                            center*8 + right*4 + bottom_left*1+ bottom*4 + bottom_right*1)/28

                '''
                    Performing Mean blur
                    kernel= | 1  1  1 |
                            | 1  1  1 |
                            | 1  1  1 |
                '''
                blur01[x,y] = (top_left *1 + top*1 + top_right*1 + left*1 +
                            center*1 + right*1 + bottom_left*1 + bottom*1 + bottom_right*1)/9

                '''
                    Sober Operation for edge detection on x-axis
                    kernel= | -1  0  1 |
                            | -2  0  2 |
                            | -1  0  1 |
                '''
                gx = (top_left * (-1) + top * 0 + top_right * 1 + left * (-2) +
                            center*0 + right*2 + bottom_left*(-1)+ bottom*0 + bottom_right*1)
                
                '''
                    Sober Operation for edge detection on y-axis
                    kernel= | -1  -2  -1 |
                            |  0   0   0 |
                            |  1   2   1 |
                '''
                gy = (top_left *(-1) + top*(-2) + top_right*(-1) + left*0 +
                            center*0 + right*0 + bottom_left*1 + bottom*2 + bottom_right*1)

                '''
                    Sober Operation for edge detection
                    (New Image) S = sqrt(Gx**2 + Gy**2)
                '''
                sober_img[x,y] = np.sqrt(gx**2 + gy**2)

                

        bcolor = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)*255
        cv.imshow("Edge", img)
        cv.imshow("GRAY", gray_img)
        cv.imshow("Gaussian Blur", blur)
        cv.imshow("Mean Blur", blur01)
        cv.imshow("Sober Edge", sober_img)

        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

    def createImage(self, data):
        response = data
        string = response['img']
        jpg_original = base64.b64decode(string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv.imdecode(jpg_as_np, flags=1)

        cv.imshow("RELOAD", img)

        key= cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()

