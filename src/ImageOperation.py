import collections
import enum
from typing import Collection
import cv2 as cv
import math
import base64
import numpy as np
from matplotlib import pyplot as plot
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag

class ImageOperation:
    def __init__(self, path):
        self.path = path
        
        self.top_left = 0
        self.top = 0
        self.top_right = 0
        self.left = 0
        self.right = 0
        self.bottom_left = 0
        self.bottom_right = 0
        self.bottom = 0
        self.center = 0
        self.colored_image= False
        print(f"Searching image at {self.path}.")
    
    def getConversionTypes(self):
        dict={
            "ConversionTypes" : ["RED", "BLUE", "GREEN", "BLUR", "GRAY", "BGR"]
        }
        return dict
    
    def readImage(self):
        return cv.imread(self.path)

    def Convert2Gray(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def showImage(self, img, frameName="Default"):
        cv.imshow(frameName, img)

    def ResizeImage(self, image, height, inter = cv.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        #r = height / float(h)
        dim = (height , height)
        resized = cv.resize(image, dim, interpolation = inter)
        return resized
    
    def cropTo(self, img):
        size = 28
        height, width = img.shape[:2]

        sideCrop = (width - 28) // 2
        return img[:,sideCrop:(width - sideCrop)]

    def getImageMat(self, color):
        img = cv.imread(self.path)
        width, height, channel= img.shape

        image = np.zeros([width, height, channel], dtype=np.uint8)
        if color=='BLUE':
            image[:,:] = img[:,:,0],0,0

            data = base64.b64encode(cv.imencode('.jpg', image)[1]).decode()
        
        elif color == 'RED':
            image[:,:] = 0,img[:,:,1],0
            data = base64.b64encode(cv.imencode('.jpg', image)[1]).decode()
        
        elif color == 'GREEN':
            image[:,:] = 0,0,img[:,:,2] 
            data = base64.b64encode(cv.imencode('.jpg', image)[1]).decode()
           
        elif color == 'GRAY':
            gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            data = base64.b64encode(cv.imencode('.jpg', gray_image)[1]).decode()
        
        elif color == 'BLUR':
            data = self.BlurImage()
        
        else:
            color = 'BGR'
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

    def convertToBW(self, img, threshold=127):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        width, height = img.shape
        bw= np.zeros((width, height))

        for x in range(0, width):
            for y in range(0, height):
                if img[x,y] < threshold:
                    bw[x,y] = 255
                else:
                    bw[x,y] = 0
                
        return bw

        '''b0 = (img >> 0) & 1
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
        plot.hist(StretchHistImg.ravel(), 255, [0, 255])'''
        '''plot.hist(logImg.ravel(), 255, [0, 255])
        plot.hist(powLowImg.ravel(), 255, [0, 255])
        plot.hist(powHighImg.ravel(), 255, [0, 255])'''

        '''plot.show()

        key = cv.waitKey(0)
        if key == 27:
            cv.destroyAllWindows()'''

    def LogTransformation(self, img, constant= 20):
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        log_img = img.copy()

        width, height= img.shape

        for x in range(0, width):
            for y in range(0, height):
                #Logarithmic Transformation -> S=c*log(1+r)
                log_img[x,y] = constant * math.log(1+img[x,y])

        return log_img

    def PowerLawTransformation(self, img, gama=0.5, constant= 20):
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        power_law_img = img.copy()

        width, height= img.shape

        for x in range(0, width):
            for y in range(0, height):
                #Power law Transformation -> S=c * pow(r, gama)
                power_law_img[x,y] = constant * math.pow(img[x,y], gama)
                
        return power_law_img

    def DigitalNegative(self, img):
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        width, height = img.shape
        negative_img= img.copy()

        for x in range(0, width):
            for y in range(0, height):
                #img is converted to Digital Negative image
                scaling = img[x,y]/255
                dn = 1-scaling
                rescaling = int(dn * 255)
                negative_img[x,y] = rescaling

        return negative_img

    def PaddingImage(self, img, x, y, height, width, channel):
        #   _________________________________
        #   | x-1, y-1 | x, y-1 | x+1, y-1  |
        #   | x-1, y   | x, y   | x+1, y    |
        #   | x-1, y+1 | x, y+1 | x+1, y+1  |
        #   |__________|________|___________| 

        if x == 0 and y == 0:
            self.top         = 0
            self.top_left    = 0 
            self.top_right   = 0
            self.left        = 0
            self.bottom_left = 0
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
            self.bottom      = img[x, y+1, channel] if self.colored_image else img[x,y+1]
            self.bottom_right= img[x+1,y+1, channel] if self.colored_image else img[x+1,y+1]

        elif x == 0 and y == height:
            self.top_left    = 0 
            self.left        = 0
            self.bottom_left = 0
            self.bottom      = 0
            self.bottom_right= 0
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_right   = img[x+1,y-1, channel] if self.colored_image else img[x+1,y-1]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
        
        elif x == width and y == height:
            self.top_right   = 0 
            self.right       = 0
            self.bottom_left = 0
            self.bottom      = 0
            self.bottom_right= 0
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_left    = img[x-1,y-1, channel] if self.colored_image else img[x-1,y-1] 
            self.left        = img[x-1,y, channel] if self.colored_image else img[x-1,y]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]

        elif y == 0:
            self.top         = 0
            self.top_left    = 0 
            self.top_right   = 0
            self.left        = img[x-1,y, channel] if self.colored_image else img[x-1,y]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
            self.bottom_left = img[x-1,y+1, channel] if self.colored_image else img[x-1,y+1]
            self.bottom      = img[x,y+1, channel] if self.colored_image else img[x, y+1]
            self.bottom_right= img[x+1,y+1,channel] if self.colored_image else img[x+1,y+1]
        
        elif x == 0:
            self.left        = 0
            self.top_left    = 0
            self.bottom_left = 0
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_right   = img[x+1,y-1,channel] if self.colored_image else img[x+1,y-1]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
            self.bottom      = 0
            self.bottom_right= 0

        elif x == width:
            self.top_right   = 0
            self.bottom_right= 0 
            self.right       = 0
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_left    = img[x-1,y-1, channel] if self.colored_image else img[x-1,y-1] 
            self.left        = img[x-1,y, channel] if self.colored_image else img[x-1,y]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.bottom_right= img[x+1,y+1,channel] if self.colored_image else img[x+1,y+1]
            self.bottom      = img[x,y+1, channel] if self.colored_image else img[x, y+1]

        elif y == height:
            self.bottom_left = 0
            self.bottom      = 0
            self.bottom_right= 0
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_left    = img[x-1,y-1, channel] if self.colored_image else img[x-1,y-1] 
            self.top_right   = img[x+1,y-1, channel] if self.colored_image else img[x+1,y-1]
            self.left        = img[x-1,y, channel] if self.colored_image else img[x-1,y]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
        
        else:
            self.top         = img[x,y-1,channel] if self.colored_image else img[x,y-1]
            self.top_left    = img[x-1,y-1, channel] if self.colored_image else img[x-1,y-1] 
            self.top_right   = img[x+1,y-1, channel] if self.colored_image else img[x+1,y-1]
            self.left        = img[x-1,y, channel] if self.colored_image else img[x-1,y]
            self.center      = img[x, y, channel] if self.colored_image else img[x,y]
            self.right       = img[x+1, y, channel] if self.colored_image else img[x+1,y]
            self.bottom_right= img[x+1,y+1,channel] if self.colored_image else img[x+1,y+1]
            self.bottom      = img[x,y+1, channel] if self.colored_image else img[x, y+1]
            self.bottom_left = img[x-1,y+1, channel] if self.colored_image else img[x-1,y+1]

    def GaussianBlur(self, img):
        blur = img.copy()

        if len(img.shape)==3:
            print("Color image")
            self.colored_image= True
            width, height, channel= img.shape
        
        else:
            print("Gray scale")
            self.colored_image= False
            width, height = img.shape

        '''
            Performing Gaussian blur
            kernel= | 1  2  1 |
                    | 2  4  2 |
                    | 1  2  1 |
        '''
        for x in range(0, width-1):
            for y in range(0, height-1):
                if self.colored_image:
                    for c in range(0, channel):
                        self.PaddingImage(img, x, y, height=height, width=width, channel=c)   

                        new_value= int((self.top_left * 1 + self.top * 4 + self.top_right*1 + self.left*4 + 
                                    self.center*8 + self.right*4 + self.bottom_left*1 + self.bottom*4 + self.bottom_right*1)/28)
                        
                        #print(f"Channel{c}=>{new_value}")
                        blur[x,y,c] = new_value
                        
                else:
                    self.PaddingImage(img, x, y, height=height, width=width, channel=0)                
                    
                    blur[x,y] = (self.top_left * 1 + self.top *4 + self.top_right*1 + self.left*4 +
                                self.center*8 + self.right*4 + self.bottom_left*1+ self.bottom*4 + self.bottom_right*1)/28
                                               
        return blur       


    def MeanBlur(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = gray_img.copy()
        
        width, height = gray_img.shape

        for x in range(0, width-1):
            for y in range(0, height-1):
                self.PaddingImage(gray_img, x, y, height=height, width=width)  
                '''
                    Performing Mean blur
                    kernel= | 1  1  1 |
                            | 1  1  1 |
                            | 1  1  1 |
                '''
                blur[x,y] = (self.top_left *1 + self.top*1 + self.top_right*1 + self.left*1 +
                            self.center*1 + self.right*1 + self.bottom_left*1 + self.bottom*1 + self.bottom_right*1)/9
        
        return blur    

    def SharpenImage(self, img):
        gray_img = img
        sharpen_img = gray_img.copy()
        
        width, height = gray_img.shape

        for x in range(0, width-1):
            for y in range(0, height-1):
                self.PaddingImage(gray_img, x, y, height=height, width=width)  
                
                '''
                    Performing Shapen
                    kernel= | 0  1  0 |
                            | 1 -4  1 |
                            | 0  1  0 |
                '''
                sharpen_img[x,y] = (self.top_left * 0 + self.top * 1 + self.top_right * 0 + self.left * 1 +
                            self.center * (-4) + self.right * 1 + self.bottom_left * 0 + self.bottom*1 + self.bottom_right*0)
                        
        
        return sharpen_img 

    def SoberEdge(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #gray_img = self.GaussianBlur()
        sober_edge_img = gray_img.copy()
        
        width, height = gray_img.shape

        for x in range(0, width-1):
            for y in range(0, height-1):
                self.PaddingImage(gray_img, x, y, height=height, width=width)    
                '''
                    Sober Operation for edge detection on x-axis
                    kernel= | -1  0  1 |
                            | -2  0  2 |
                            | -1  0  1 |
                '''
                gx = (self.top_left * (-1) + self.top * 0 + self.top_right * 1 + self.left * (-2) +
                           self.center*0 + self.right*2 + self.bottom_left*(-1)+ self.bottom*0 + self.bottom_right*1)
                
                '''
                    Sober Operation for edge detection on y-axis
                    kernel= | -1  -2  -1 |
                            |  0   0   0 |
                            |  1   2   1 |
                '''
                gy = (self.top_left *(-1) + self.top*(-2) + self.top_right*(-1) + self.left*0 +
                            self.center*0 + self.right*0 + self.bottom_left*1 + self.bottom*2 + self.bottom_right*1)

                '''
                    Sober Operation for edge detection
                    (New Image) S = sqrt(Gx**2 + Gy**2)
                '''
                sober_edge_img[x,y] = np.sqrt(gx**2 + gy**2)

        return sober_edge_img
    
    def Dilation(self, img):
        kernel = np.array([255.0,255.0,255.0])
        newImg01 = img.copy()

        print(kernel)
        print(kernel.shape)
        cv.imshow("BINARY IMAGE", img)

        w,h = img.shape

        class Match(enum.Enum):
            all_match= 1
            some_match= 1
            no_match= 0

        for x in range(0, w-1):
            for y in range(0, h-1):
                self.PaddingImage(img, x, y, h, w)
                if (self.left == kernel[0] and self.center == kernel[1] and self.right==kernel[2] 
                        and self.top_right==kernel[0] and self.top_left == kernel[0] and self.top == kernel[1] 
                        and self.bottom==kernel[2] and self.bottom_left==kernel[2] and self.bottom_right==kernel[2]):
                    newImg01[x,y]= 255.0
                elif (self.left == kernel[0] or self.center == kernel[1] or self.right==kernel[2] 
                        or self.top_right==kernel[0] or self.top_left == kernel[0] or self.top == kernel[1] 
                        or self.bottom==kernel[2] or self.bottom_left==kernel[2] or self.bottom_right==kernel[2]):
                    newImg01[x,y]= 255.0
                else:
                    newImg01[x,y]= 0.0

        return newImg01
    
    def Erosion(self, img):
        kernel = np.array([255.0,255.0,255.0])
        newImg01 = img.copy()

        w,h = img.shape

        class Match(enum.Enum): 
            all_match= 1
            some_match= 1
            no_match= 0

        for x in range(0, w-1):
            for y in range(0, h-1):
                self.PaddingImage(img, x, y, h, w)
                if (self.left == kernel[0] and self.center == kernel[1] and self.right==kernel[2] 
                        and self.top_right==kernel[0] and self.top_left == kernel[0] and self.top == kernel[1] 
                        and self.bottom==kernel[2] and self.bottom_left==kernel[2] and self.bottom_right==kernel[2]):
                    newImg01[x,y]= 255.0
                elif (self.left == kernel[0] or self.center == kernel[1] or self.right==kernel[2] 
                        or self.top_right==kernel[0] or self.top_left == kernel[0] or self.top == kernel[1] 
                        or self.bottom==kernel[2] or self.bottom_left==kernel[2] or self.bottom_right==kernel[2]):
                    newImg01[x,y]= 0.0
                else:
                    newImg01[x,y]= 0.0

        
        return newImg01

    def CNNFromScratch(self, img, filter_size=3, channel=1):
        filter= np.random.randn(channel, filter_size, filter_size)/(filter_size * filter_size)

        w,h= img.shape
        feature_map= np.zeros((w-filter_size+1, h-filter_size+1, channel))
        for y in range(h-filter_size+1):
            for x in range(w-filter_size+1):
                feature_map[x,y]= np.sum(img[x:x+filter_size, y:y+filter_size] * filter, axis=(1,2))

        return feature_map
    
    def MaxPooling(self, image, filter_size=2, channel=1):
        new_height= image.shape[1] // filter_size
        new_width= image.shape[0] // filter_size
        
        max_pool= np.zeros((new_width, new_height, channel))

        for y in range(new_height):
            for x in range(new_width):
                max_pool[x,y]= np.amax(image[x*filter_size:x*filter_size+filter_size,y*filter_size:y*filter_size+filter_size], axis=(0,1))

        return max_pool

    def CreateStack(self, pool=()):
        stack_struct= []
        for col in range(len(pool)):
            '''for im in range(len(pool[col])):
                pool[col][im] = self.ResizeImage(pool[col][im], scale=(1/2))
'''
            stack_struct.append(np.hstack(pool[col]))
        
        img= np.vstack(stack_struct)

        return img

        #return img

    def Convolution(self, img):
        #img= self.SoberEdge(img)
        if len(img.shape)==3:
            print("Color image")
            self.colored_image= True
            width, height, channel= img.shape
        
        else:
            print("Gray scale")
            self.colored_image= False
            width, height = img.shape
            
        newImg= img.copy()

        for x in range(0, width-1):
            for y in range(0, height-1):
                if self.colored_image:
                    for c in range(0, channel):
                        self.PaddingImage(img, x, y, height, width, c)
                        value= (self.top_left *1 + self.top*0 + self.top_right*1+ self.left*0 +
                            self.center*1 + self.right*0 + self.bottom_left*1 + self.bottom*0 + self.bottom_right*1)
        
                        try:
                            newImg[x,y,c]= value
                        except:
                            print(f"Tried newImg[{x}, {y}]:: {value}")
                else:
                    self.PaddingImage(img, x, y, height, width, 0)

                    value= (self.top_left *1 + self.top*0 + self.top_right*1+ self.left*0 +
                            self.center*1 + self.right*0 + self.bottom_left*1 + self.bottom*0 + self.bottom_right*1)
        
                    try:
                        newImg[x,y]= value
                    except:
                        print(f"Tried newImg[{x}, {y}]:: {value}")

        return newImg

    def getImagePixel(self, img, color):
        image = np.asarray(bytearray(img), dtype="uint8")
        image = cv.imdecode(image, cv.COLOR_RGB2BGR)
        width, height, channel = image.shape
        print(width)
        print(height)
        print(channel)
        computedImg = image.copy()

        if color=='BLUE':
            computedImg[:,:,0] = image[:,:,0]
            computedImg[:,:,1] = 0
            computedImg[:,:,2] = 0
  
        elif color == 'RED':
            computedImg[:,:,2] = image[:,:,2]
            computedImg[:,:,1] = 0
            computedImg[:,:,0] = 0

        elif color == 'GREEN':
            computedImg[:,:,1] = image[:,:,1] 
            computedImg[:,:,2] = 0
            computedImg[:,:,0] = 0

        elif color == 'GRAY':
            computedImg = (image[:,:,0],image[:,:,1],image[:,:,2])/3
        
        elif color == 'BLUR':
            computedImg = self.BlurImage(image)
        
        else:
            color = 'BGR'
            computedImg[:,:] = image[:,:]
           
        #This is responsible to convert the image in list format to string
        img_encode = cv.imencode('.jpg', computedImg)[1]

        data_encode = np.array(img_encode)
        str_encode = data_encode.tostring()
        
        return str_encode

    def CreateTrackbar(self, trackbar_name="Value",
                            title="Default", 
                            size=[300,150], 
                            min= 0, max= 255):
        cv.namedWindow(title)
        cv.resizeWindow(title, size[0], size[1])
        cv.createTrackbar(trackbar_name, title, min, max, self.UpdateFrame)

    def MaskImage(self):
        original_img = cv.imread(self.path)
        w,h,channel = original_img.shape

        '''dim = [int(w/1.9),int(h/3)]
        original_img= cv.resize(original_img, dim, interpolation= cv.INTER_AREA)'''

        hsv_img = cv.cvtColor(original_img, cv.COLOR_BGR2HSV)

        horizontal_combine = np.hstack((original_img, original_img, original_img))

        trackbar_title= "Controller"
        self.CreateTrackbar(trackbar_name="HueMin", title=trackbar_title, min= 0, max= 179)
        self.CreateTrackbar(trackbar_name="HueMax", title=trackbar_title, min= 68, max= 179)
        self.CreateTrackbar(trackbar_name="SatMin", title=trackbar_title, min= 36, max= 255)
        self.CreateTrackbar(trackbar_name="SatMax", title=trackbar_title, min= 180, max= 255)
        self.CreateTrackbar(trackbar_name="VMin", title=trackbar_title, min= 137, max= 255)
        self.CreateTrackbar(trackbar_name="VMax", title=trackbar_title, min= 255, max= 255)

        cv.imshow("Original", original_img)
        cv.imshow("HSV Image", hsv_img)
        cv.imshow("HORIZONTAL Image", horizontal_combine)

        newImg= original_img.copy()

        while True:
            hue_min= cv.getTrackbarPos("HueMin", trackbar_title)
            hue_max= cv.getTrackbarPos("HueMax", trackbar_title)
            sat_min= cv.getTrackbarPos("SatMin", trackbar_title)
            sat_max= cv.getTrackbarPos("SatMax", trackbar_title)
            v_min= cv.getTrackbarPos("VMin", trackbar_title)
            v_max= cv.getTrackbarPos("VMax", trackbar_title)

            lower = np.array([hue_min, sat_min, v_min])
            upper = np.array([hue_max, sat_max, v_max])

            mask = cv.inRange(hsv_img, lower, upper)
            newImg = cv.bitwise_and(original_img, original_img, mask=mask)

            cv.imshow("Mask", mask)
            cv.imshow("NewImg", newImg)

            self.WaitForClosing();

    #Holding the image window until user press "ESC"- key
    def WaitForClosing(self):
        key= cv.waitKey(0) & 0xFF
        if key == 27:
            cv.destroyAllWindows()

    def DetermineContours(self):
        objectPoints=[]
        video = cv.VideoCapture(0)
        while True:
            success, img = video.read()
            imgContour = img.copy()
            blur= cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (5,5), cv.BORDER_DEFAULT)
            edge=cv.Canny(blur, 50, 50)

            cv.imshow("Original", img)
            cv.imshow("Blur", blur)
            cv.imshow("Edge", edge)

            contours, hierarchy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            for con in contours:
                area = cv.contourArea(con)

                cv.drawContours(imgContour, con, 0, (150,120,0), 2)

                arcLength = cv.arcLength(con, True)
                totalNodes = cv.approxPolyDP(con, 0.02*arcLength, True)

                x,y,w,h = cv.boundingRect(totalNodes)

                objectPoints.append([x,y,w,h])

                cv.rectangle(imgContour, (x,y), (x+w,y+h), (0,255,0), 2)

            cv.imshow("Contour", imgContour)
            
            self.WaitForClosing()

    def StackImages(self, img=()):
        combined_image = np.hstack(img)

        return combined_image

    def ImageSegmentation(self, image):
        plot.hist(image.ravel(), 255, [0,255])
        plot.show()

    def GlobalThresholding(self, image):
        w,h= image.shape

        for x in range(0, w):
            for y in range(0, h):
                if image[x,y]<100:
                    image[x,y]=0
                elif image[x,y]>=100 and image[x,y]<200:
                    image[x,y]= 200
                else:
                    image[x,y]= 255
                
        return image
                

