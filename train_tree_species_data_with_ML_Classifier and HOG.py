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

modelname = 'modelname.sav'

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="training\\")
ap.add_argument("-t", "--test", required=True, help="test1\\")
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
 
	# load the image, convert it to grayscale, and detect edges
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	H = feature.hog(gray, pixels_per_cell=(16, 16),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
 
	# update the data and labels
	data.append(H)
	labels.append(make)

##print(labels)
# "train" the nearest neighbors classifier
print("[INFO] training classifier...")

X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.80,test_size = 0.2)

print(len(X_train), len(X_test),len(y_train), len(y_test))
print("for banana")
print(len(data[0]))

##model = DecisionTreeClassifier(criterion='gini',random_state=9)
#max_features='log2')
##model = RandomForestClassifier(n_estimators=100)
##model = KNeighborsClassifier(n_neighbors=3)
model = SVC(kernel='linear', probability=True)
##model = SVC(kernel = 'sigmoid')
##model = AdaBoostClassifier(n_estimators=90)
##logreg = LogisticRegression()
##model=RFE(logreg,10)
##model =  GaussianNB()


model.fit(X_train, y_train)


print("[INFO] evaluating...")
pickle.dump(model, open(modelname, 'wb'))
y_pred = model.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)



##y_score = model.fit(X_train, y_train).predict_proba(X_test)
##fpr = dict()
##tpr = dict()
##roc_auc = dict()
##for i in range(6):
##    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
##    roc_auc[i] = auc(fpr[i], tpr[i])
##colors = cycle(['blue', 'red', 'green','yellow','black','orange'])
##for i, color in zip(range(6), colors):
##    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
##             label='ROC curve of class {0} (area = {1:0.2f})'
##             ''.format(i, roc_auc[i]))
##
##plt.plot([0, 1], [0, 1], 'k--', lw=lw)
##plt.xlim([-0.05, 1.0])
##plt.ylim([0.0, 1.05])
##plt.xlabel('False Positive Rate')
##plt.ylabel('True Positive Rate')
##plt.title('Receiver operating characteristic for multi-class data')
##plt.legend(loc="lower right")
##plt.show()






##text_representation = tree.export_text(model)
##print(text_representation)
##fig = plt.figure(figsize=(25,20))
##_ = tree.plot_tree(model, 
##                   feature_names=data,  
##                   class_names=labels,
##                   filled=True)
##
##fig.savefig("decistion_tree.png")
##
##
##dot_data = StringIO()
##export_graphviz(model, out_file=dot_data,  
##                filled=True, rounded=True,
##                special_characters=True,feature_names = data,class_names=['Banana','Coconut','Mango','Not specified','Palm','Papaya'])
##graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
##graph.write_png('decision-tree1.png')
##Image(graph.create_png())

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


