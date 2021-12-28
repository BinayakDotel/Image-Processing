import pickle as pk
from tensorflow.keras.models import load_model
import numpy as np

class ImageClassifier:
    def __init__(self):
        self.item_names= ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    def predict(self, image):
        model= load_model('models\models\my_model')
        output= self.item_names[np.argmax(model.predict(image))]
        print(output)