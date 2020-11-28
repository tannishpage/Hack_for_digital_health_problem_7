from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return "." in filename and filename.rsplid(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"Error" : "No File Passed"})
        if not allowed_file(file.filename):
            return jsonify({"Error" : "File format not supported"})
        
        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {"Prediction": prediction.item(), "class_name": "Benign" if prediction.item() == 0 else "Malignent"}
            return data
        except Exception as e:
            return jsonify({"Error" : e})


if __name__ == "__main__":
    app.run(port=8080)