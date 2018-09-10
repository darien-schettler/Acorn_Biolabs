import imutils
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

"""

Before we can recognize faces in images and videos, we first need to quantify the faces in our training set. Keep
in mind that we are not actually training a network here — the network has already been trained to create 128-d
embeddings on a dataset of ~3 million images (resnet-34)

We certainly could train a network from scratch or even fine-tune the weights of an existing model but that is more
than likely overkill for many projects. Furthermore, you would need a lot of images to train the network from scratch.

Instead, it’s easier to use the pre-trained network and then use it to construct 128-d embeddings for each of the 218
faces in our dataset.

Then, during classification, we can use a simple k-NN model + votes to make the final face classification. Other
traditional machine learning models can be used here as well

"""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", default="dataset",
                help="path to input directory of faces + images")

ap.add_argument("-e", "--encodings", default="encodings.pickle",
                help="path to serialized db of facial encodings")

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())

'''

Let’s list out the argument flags and discuss them:

--dataset           : The path to our data-set 
--encodings         : Our face encodings are written to the file that this argument points to
--detection-method  : Before we can encode faces in images we first need to detect them
                        --> Two face detection methods include either hog  or cnn
                        --> Those two flags are the only ones that will work for --detection-method

'''

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(imutils.paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

