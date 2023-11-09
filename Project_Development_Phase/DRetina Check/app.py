from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
import os
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
model = load_model(r"inception-diabetic.h5", compile=False)
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def upload():
    try:
        if request.method == 'POST':
            f = request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)

            print("File saved to:", filepath)  # Add this line

            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            pred = np.argmax(model.predict(x), axis=1)
            index = [
                "No apparent Diabetic retinopathy",
                "Mild nonproliferative Diabetic retinopathy (Mild NPDR)",
                "Moderate nonproliferative Diabetic retinopathy (Moderate NPDR)",
                "Severe nonproliferative Diabetic retinopathy (Severe NPDR)",
                "Proliferative Diabetic retinopathy (PDR)",
            ]
            # Additional information about each class
            class_info = {
                "No apparent Diabetic retinopathy": "As you have no apparent retinopathy, manage your diabetes by, Keeping blood sugar levels, blood pressure, and cholesterol levels under control through lifestyle changes, medication, and regular monitoring is essential to prevent the development of diabetic retinopathy.",
                "Mild nonproliferative Diabetic retinopathy (Mild NPDR)": "As you have Mild nonproliferative Diabetic retinopathy (Mild NPDR), Continue with strict blood sugar control.Monitor blood pressure and cholesterol levels. Regular eye exams, usually annually.",
                "Moderate nonproliferative Diabetic retinopathy (Moderate NPDR)": "As you have Moderate nonproliferative Diabetic retinopathy (Moderate NPDR), follow Intensify blood sugar control. Manage blood pressure and cholesterol.Consider additional interventions based on the eye care professional's recommendation. More frequent eye exams, potentially every 6 to 12 months.",
                "Severe nonproliferative Diabetic retinopathy (Severe NPDR)": "As you have Severe nonproliferative Diabetic retinopathy (Severe NPDR), follow Aggressive management of diabetes, blood pressure, and cholesterol. Laser treatment (photocoagulation) to reduce swelling. Possible referral to a retina specialist. More frequent eye exams, typically every 3 to 6 months.",
                "Proliferative Diabetic retinopathy (PDR)": "As you have Proliferative Diabetic retinopathy (PDR), follow Laser surgery (photocoagulation) to shrink abnormal blood vessels. Intravitreal injections of medications to reduce abnormal vessel growth. Vitrectomy surgery if there is bleeding into the vitreous. Regular and frequent eye exams, possibly every 2 to 4 months.",
            }
            predicted_class = index[pred[0]]
            result_text = f"The Predicted Class is: {predicted_class}\n\nSuggested medication:\n{class_info[predicted_class]}"
            print("Prediction result:", result_text)

            return result_text
    except Exception as e:
        print(f"Error: {e}")
        return 'Error predicting. Please try again.'

if __name__ == '__main__':
    app.run(debug=True)
