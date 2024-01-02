from flask import Flask, render_template, request
import numpy as np
import re
import base64
from PIL import Image
from imageio import imread
# from scipy.misc import imread, imresize
from keras.models import load_model
import json

app = Flask(__name__)

conv = load_model("./models/")