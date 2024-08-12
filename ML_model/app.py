from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_cancer():
    radius_mean = float(request.form.get('radius_mean'))
    perimeter_mean = float(request.form.get('perimeter_mean'))
    area_mean = float(request.form.get('area_mean'))
    compactness_mean = float(request.form.get('compactness_mean'))
    concavity_mean = float(request.form.get('concavity_mean'))
    concave_points_mean = float(request.form.get('concave points_mean'))
    radius_se = float(request.form.get('radius_se'))
    perimeter_se = float(request.form.get('perimeter_se'))
    area_se = float(request.form.get('area_se'))
    radius_worst = float(request.form.get('radius_worst'))
    texture_worst = float(request.form.get('texture_worst'))
    perimeter_worst = float(request.form.get('perimeter_worst'))
    area_worst = float(request.form.get('area_worst'))
    compactness_worst = float(request.form.get('compactness_worst'))
    concavity_worst = float(request.form.get('concavity_worst'))
    concave_points_worst = float(request.form.get('concave points_worst'))

    result = model.predict(np.array([radius_mean,perimeter_mean,area_mean,compactness_mean,concavity_mean,concave_points_mean,radius_se,perimeter_se,area_se,radius_worst,texture_worst,perimeter_worst,area_worst,compactness_worst,concavity_worst,concave_points_worst]).reshape(1,16))
    if result[0] == 0:
        result = 'Breast cancer type is Benign'
    else:
        result = 'Breast cancer type is Malignant'

    return render_template('index.html',result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)