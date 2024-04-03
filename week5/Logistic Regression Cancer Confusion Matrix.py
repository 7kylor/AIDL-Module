#Import libraries
import pandas as pd

#Read the csv file to a dataframe
df = pd.read_csv('cancer.csv')

#Show the dataframe
df.head(5)

#Separate the features 
features = df.drop('diagnosis', axis = 'columns')

#Separate the target
target = df['diagnosis']

#Assigning to conventional variables, the features and target
X_train = features
Y_train = target

#Split the datset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.3, random_state=0)
#random_state = 0; we get the same train and test sets across different executions

#Print the dimension of train and test data
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

## Logic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train Logistic Regression model
model.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = model.predict(X_test)

# Model Accuracy, irrespectove of benign or malignant?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

#Get the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

TN = cm[0][0]
FN = cm[1][0]
FP = cm[0][1]
TP = cm[1][1]

print ('TP =', TP)
print ('TN =', TN)
print ('FP =', FP)
print ('FN =', FN)

#print confusion matrix plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, Y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

#print specificity
print('Specificity = ', TN / (TN + FP))

#Print sensitivity
print('Sensitivity = ', TP / (TP + FN))

# print Precision
from sklearn import metrics
print("Precision:",metrics.precision_score(Y_test, Y_pred, average = None))

#print Recall
from sklearn import metrics
print("Recall:",metrics.recall_score(Y_test, Y_pred, average = None))

# print f1 score
from sklearn import metrics
print("F-score:",metrics.f1_score(Y_test, Y_pred, average = None))

#print classification report 
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))