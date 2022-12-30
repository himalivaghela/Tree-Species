# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO
##from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import pylab as pl
from sklearn.naive_bayes import GaussianNB
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="training\\")
ap.add_argument("-t", "--test", required=True, help="test1\\")
args = vars(ap.parse_args())



filename ='modelname.sav'
model = pickle.load(open(filename, 'rb'))
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
	# load the test image, convert it to grayscale, and resize it to
	# the canonical size
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#logo = cv2.resize(gray, (200, 100))
 
	# extract Histogram of Oriented Gradients from the test image and
	# predict the make of the car
	(H, hogImage) = feature.hog(gray,  pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualize=True)
	pred = model.predict(H.reshape(1, -1))[0]
 
	# visualize the HOG image
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
##	cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
 
	# draw the prediction on the test image and display it
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	cv2.imshow("Test Image #{}".format(i + 1), image)
	cv2.waitKey(0)

