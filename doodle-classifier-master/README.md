# doodle-classifier
Inspired By SouravJohar
## Requirements
* Python
* Electron.js
* node.js
* Tensorflow
* Keras
* numpy
* scikit-learn
* PIL
* scipy
* Flask


Install electron (You'll need node.js)
```sh
$ npm install electron -g
```

Download and move into the project directory
```
$ git clone https://github.com/SouravJohar/doodle-classifier.git
$ cd doodle-classifier 
```

Download the dataset inside a 'data/' directory.
Dataset : https://github.com/googlecreativelab/quickdraw-dataset (get the numpy bitmaps)

```sh
$ python train.py
```

After training, link your saved model in `server.py` and then run the server

```sh
$ python server.py
```
After the server is up and running,

```sh
$ electron .
```
