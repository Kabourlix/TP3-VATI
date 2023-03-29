from tensorflow import keras
import keras.utils as image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_test = image.load_img('C:/Users/annie/Documents/GitHub/TP3-VATI/test/ace of clubs/005.jpg',target_size=(150,150))
img_test = np.asarray(img_test)
plt.imshow(img_test)
img_test = np.expand_dims(img_test,axis=0)

# Utiliser model # ça ne marche pas encore
saved_model = load_model(r'vggclf.h5')
output = saved_model.predict(img_test)
if(output[0][0]>output[0][1]):
    print("ace of clubs")
else:
    print("not ace of clubs")