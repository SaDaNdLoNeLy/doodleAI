# from flask import Flask, render_template, request
# import numpy as np
# import re
# import base64
# from PIL import Image
# from scipy.misc import imsave, imread, imresize
# from keras.models import load_model
# from prepare_data import normalize
# import json

# app = Flask(__name__)

# # mlp = load_model("./models/mlp_94.h5")
# conv = load_model("./models/doodle.h5")
# SHAPES = {0: 'triangle', 1: 'circle', 2: 'square'}


# @app.route("/", methods=["GET", "POST"])
# def ready():
#     if request.method == "GET":
#         return render_template("index1.html")
#     if request.method == "POST":
#         data = request.form["payload"].split(",")[1]
#         net = request.form["net"]

#         img = base64.decodestring(data)
#         with open('temp.png', 'wb') as output:
#             output.write(img)
#         x = imread('temp.png', mode='L')
#         # resize input image to 28x28
#         x = imresize(x, (28, 28))

#         # if net == "MLP":
#         #     model = mlp
#         #     # invert the colors
#         #     x = np.invert(x)
#         #     # flatten the matrix
#         #     x = x.flatten()

#         #     # brighten the image a bit (by 60%)
#         #     for i in range(len(x)):
#         #         if x[i] > 50:
#         #             x[i] = min(255, x[i] + x[i] * 0.60)

#         if net == "ConvNet":
#             model = conv
#             x = np.expand_dims(x, axis=0)
#             x = np.reshape(x, (28, 28, 1))
#             # invert the colors
#             x = np.invert(x)
#             # brighten the image by 60%
#             for i in range(len(x)):
#                 for j in range(len(x)):
#                     if x[i][j] > 50:
#                         x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

#         # normalize the values between -1 and 1
#         x = normalize(x)
#         val = model.predict(np.array([x]))
#         pred = SHAPES[np.argmax(val)]
#         classes = ["Apple", "Banana", "Grape", "Pineapple"]
#         print (pred)
#         print( list(val[0]))
#         return render_template("index1.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"], net=net)


# app.run()

from flask import Flask, render_template, request
import numpy as np
import re
import base64
from PIL import Image
from imageio import imread, imwrite
from keras.models import load_model
from prepare_data import normalize
import json

app = Flask(__name__)

# mlp = load_model("./models/mlp_94.h5")
object_file = open("./static/object.txt", "r")
objects = object_file.readlines()
object_file.close()
N_CLASSES = len(objects)
CLASSES = {}
for idx, obj in enumerate(objects):
    CLASSES[idx] = obj.replace('\n', '')
print(CLASSES)
conv = load_model("./models/doodle_2024-01-03_01-44-36.h5")
SHAPES = CLASSES

def save_numpy_array(array, file_path):
    """
    Save a NumPy array to a file.

    Parameters:
    - array: NumPy array to be saved.
    - file_path: File path to save the array to.
    """
    np.save(file_path, array)
    print(f"NumPy array saved to {file_path}")

@app.route("/", methods=["GET", "POST"])
def ready():
    if request.method == "GET":
        classes =list(SHAPES.values())
        classesText = f"Draw an object belong to one of the following categories: {', '.join(classes)}. Then see the prediction"
        return render_template("index1.html", classesText=classesText)
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        net = request.form["net"]

        img = base64.decodebytes(data.encode('utf-8'))
        with open('static/temp.png', 'wb') as output:
            output.write(img)
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = imread('temp.png', pilmode='L')
        # resize input image to 28x28
        x = np.array(Image.fromarray(x).resize((28, 28)))

        # if net == "MLP":
        #     model = mlp
        #     # invert the colors
        #     x = np.invert(x)
        #     # flatten the matrix
        #     x = x.flatten()

        if net == "CNN":
            model = conv
            x = np.expand_dims(x, axis=0)
            x = np.reshape(x, (28, 28, 1))
            # invert the colors
            x = np.invert(x)
            # brighten the image by 60%
            for i in range(len(x)):
                for j in range(len(x)):
                    if x[i][j] > 50:
                        x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        print(val)
        save_numpy_array(np.array([x]), "user-data.npy")
        pred = SHAPES[np.argmax(val)]
        classes =list(SHAPES.values())
        print(pred)
        print(classes)
        print(list(val[0]))
        classesText = f"Draw an object belong to one of the following categories: {', '.join(classes)}. Then see the prediction"
        return render_template("index1.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"], net=net, tempImage='./temp.png', classesText=classesText)

if __name__ == "__main__":
    app.run()
