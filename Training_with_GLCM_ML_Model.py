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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
import numpy as np
import mahotas
import h5py
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()

print(start_time)
# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="training\\")
ap.add_argument("-t", "--test", required=True, help="testing\\")
args = vars(ap.parse_args())
 
# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
labels = []

for imagePath in paths.list_images(args["training"]):
	# extract the make of the car
##	print(imagePath)
	make = imagePath.split("\\")[1]
##	print(imagePath)
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	H = mahotas.features.haralick(gray).mean(axis=0)
	
	# update the data and labels orientations=9 L2-Hys
	data.append(H)
	if make == 'Banana':
            print('banana')
            labels.append(0)
	elif make == 'Coconut':
            print('coconut')
            labels.append(1)
	elif make == 'Mango':
            print('Mango')
            labels.append(2)   
	elif make == 'Not specified':
            print('NS')
            labels.append(3)
	elif make == 'Oil Palm':
            print('OP')
            labels.append(4)
	elif make == 'Papaya':
            print('Papaya')
            labels.append(5)
            
        

##print(labels)
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.80,test_size = 0.20)
print(len(X_train), len(X_test), len(y_train), len(y_test))

print(type(X_test))
svmodel = SVC(kernel='linear',gamma = 0.142, probability=True) # poly, sigmoid, rbf
model = OneVsOneClassifier(svmodel)
##model = RandomForestClassifier(n_estimators=100)
##model = KNeighborsClassifier(n_neighbors=3)
##model = AdaBoostClassifier(n_estimators=90)
##logreg = LogisticRegression()
##model = DecisionTreeClassifier(criterion='gini',random_state=9)
model.fit(X_train, y_train)
print("[INFO] evaluating...")
modelname = 'glcm_rf.sav'
pickle.dump(model, open(modelname, 'wb'))

y_pred = model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)

####matrix = plot_confusion_matrix(model, X_test, y_test,
####                                 cmap=plt.cm.Blues,
####                                 normalize='true')
####plt.title('Confusion matrix for OvR classifier')
######plt.show()
####plt.show()
##
### Plot decision boundary
####X_test = np.array(X_test)
####y_test = np.array(y_test)
####plot_decision_regions(X_test.values, y_test.values, clf=model, legend=2)
####plt.show()
##
####
####end_time = time.time()
####print("Total execution time: {} seconds".format(end_time - start_time))


