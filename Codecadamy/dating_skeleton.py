import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Create your df here:
#Load data from csv file
df1 = pd.read_csv('profiles.csv')

#Height
plt.subplot(131)
plt.hist(df1.height, bins=40)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.xlim(50, 85)

#status
plt.subplot(132)
label_status = ['single', 'seeing someone', 'available', 'married', 'unknown']
plt.pie(df1.status.value_counts(), labels=label_status, autopct='%1.1f%%')
plt.title('Relationship Status')

#sex
plt.subplot(133)
labels = ['Male', 'Female']
plt.pie(df1.sex.value_counts(), labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Male vs Female')
plt.show()

#Augment data and create new columns

'''Questions I am interested in:
1. Can we predict sex of the person based on wordcount of 'My self summary' section and level of income?
2. Is there a bias in salaries? Meaning, can we predict sex of the person based on salary alone?'''

sex_mapping={'m': 0, 'f': 1}
df1['sex_code'] = df1.sex.map(sex_mapping)

df1['essay0'] = df1['essay0'].replace(np.nan, '', regex=True)
df1['essay_word_cnt'] = df1['essay0'].apply(lambda x: len(x.split()))

#normalizing data

feature_data = df1[['sex_code', 'essay_word_cnt', 'income']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

#create training and validation datasets

training_data, validation_data, training_labels, validation_labels = train_test_split(feature_data[['essay_word_cnt', 'income']], feature_data.sex_code, train_size=0.8, test_size=0.2, random_state=100)
print('Classification methods')

#Support Vector Machine
start_svm = time.time()
svm_classifier = SVC(kernel='rbf', gamma=1)
svm_classifier.fit(training_data, training_labels)
end_svm = time.time()
print('SVM score was: ', svm_classifier.score(validation_data, validation_labels), 'and took ', end_svm - start_svm, ' seconds to compute')


#Naive Bayes
start = time.time()
nBayes = MultinomialNB()
nBayes.fit(training_data, training_labels)
end = time.time()
print('Naive Bayes score was: ', nBayes.score(validation_data, validation_labels), 'and took ', end - start, ' seconds to compute')

print('Regressions')

#Simple regression - predicting sex based on level of income

linear_reg = LinearRegression()
income=np.array(training_data['income'])
income=income.reshape(-1,1)
linear_reg.fit(income, training_labels)
validation_income=np.array(validation_data['income'])
validation_income=validation_income.reshape(-1,1)
print('Predicting sex based on level of income yielded the score of :', linear_reg.score(validation_income, validation_labels))

#Multiple regression - predicting sex based on level of income and word count

multiple_reg = LinearRegression()
multiple_reg.fit(training_data, training_labels)
print('Predicting sex based on level of income and length of "About Me" essay yielded the score of :', multiple_reg.score(validation_data, validation_labels))
predictions=multiple_reg.predict(validation_data)
print(accuracy_score(validation_labels, predictions))
print(recall_score(validation_labels, predictions))
print(precision_score(validation_labels, predictions))
print(f1_score(validation_labels, predictions))
