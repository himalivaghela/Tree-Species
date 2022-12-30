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
from xgboost import XGBClassifier
from vecstack import stacking
import warnings

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
	print(make)
	if make == 'Banana':
            i = 0
	elif make == 'Coconut':
            i = 1
	elif make == 'Mango':
            i = 2 
	elif make == 'Not specified':
            i = 3
	elif make == 'Oil Palm':
            i = 4
	else:
            i = 5
        
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	H = feature.hog(gray, pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
 
	# update the data and labels
	data.append(H)
	labels.append(i)
	

            

##print(labels)
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.80,test_size = 0.2)
print(len(X_train), len(X_test),len(y_train), len(y_test))
print("for banana")
print(len(data[0]))



model1 = SVC(kernel='linear')# probability=True
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = DecisionTreeClassifier(criterion='gini',random_state=9) #max_features='log2')
model4 = RandomForestClassifier(n_estimators=100)
##model5 = GaussianNB()

models = [model1, model2, model3, model4]
warnings.filterwarnings('ignore') 
S_train, S_test = stacking(models,                   
                           X_train, y_train, X_test, regression=False, 
                           mode='oof_pred_bag', needs_proba=False,
                           save_dir=None, metric=accuracy_score, 
                           n_folds=5, stratified=True,
                           shuffle=True, random_state=0, verbose=2)





model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=3)
    
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)

##print("models parameter")
##print(model.best_params_)




