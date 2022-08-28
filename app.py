#Import necessary libraries
from flask import Flask, render_template, request, url_for

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'D:/deteksi_penyakit_tanaman_jagung/models/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_corn_diseas(corn_plant):
  test_image = load_img(corn_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return 'Bulai Daun (Downy Mildew)', 'bulai_daun_jagung.html'
       
  elif pred==1:
      return 'Daun Jagung Sehat', 'daun_jagung_sehat.html'
        
  elif pred==2:
      return 'Gejala Bercak Daun (Southern Leaf Blight)', 'gejala_bercak_daun.html'
        
  elif pred==3:
      return 'Gejala Hawar Daun (Northern Leaf Blight)', 'gejala_hawar_daun.html'
       
  elif pred==4:
      return 'Hawar Daun Komplikasi', 'hawar_daun_komplikasi.html'

    
# Create flask instance
app = Flask(__name__)

@app.route("/")
def home():
        return render_template('index.html')

@app.route("/penanganan")
def penanganan():
        return render_template('penanganan.html')

@app.route("/info")
def info():
        return render_template('info.html')

# render deteksi.html page
@app.route("/deteksi", methods=['GET', 'POST'])
def deteksi():
        return render_template('deteksi.html')
    
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('D:/deteksi_penyakit_tanaman_jagung/upload/', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_corn_diseas(corn_plant=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    







# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=2022, debug=True) 
    
    