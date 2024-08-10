from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Define flower images
flower_images = {
    'setosa': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1wrEGyVjx50qK8pxsmYOzL-KZgdZ3r2FcdA&s',
    'versicolor': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQKVrB-_-MvO3latqeOKfGjVYgG45ttVmCnuw&s',
    'virginica': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvSmAo1knGvo4I0El_KDQD99gkVSMlk2SVjA&s'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    flower_image = None
    error_message = None

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)[0]

            flower_image = flower_images.get(prediction, '')
            result = prediction
        except ValueError:
            error_message = "Please enter valid numeric values for all fields."
        except Exception as e:
            error_message = str(e)

    return render_template('index.html', result=result, flower_image=flower_image, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
