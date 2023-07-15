# Importing the  required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score, roc_curve,precision_recall_curve,auc,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pickle


# loading dataset
preprocessed_data_df=pd.read_csv("/email/dsml/final_preprocessed_data_df.csv")

new_df=preprocessed_data_df[['teacher_prefix', 'school_state','project_resource_summary', 'project_title',
       'teacher_number_of_previously_posted_projects', 'project_is_approved',
       'project_subject_categories', 'project_subject_subcategories',
       'project_grade_category', 'essay', 'price', 'quantity']]


# splitting data to X,Y and val data
from sklearn.model_selection import train_test_split
X,x_val,Y,y_val=train_test_split(
    new_df,
    new_df['project_is_approved'],
    test_size=0.2,
    random_state=42,
    stratify=preprocessed_data_df[['project_is_approved']])


print("x_train: ",X.shape)
print("x_test : ",x_val.shape)
print("y_train: ",Y.shape)
print("y_test : ",y_val.shape)

# splitting data to train and test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, stratify=Y)

print("x_train: ",x_train.shape)
print("y_train: ",y_train.shape)
print("x_val   : ",x_val.shape)
print("y_val   : ",y_val.shape)
print("x_test : ",x_test.shape)
print("y_test : ",y_test.shape)


# y_value_counts = row1['project_is_approved'].value_counts()
print("X_TRAIN-------------------------")
x_train_y_value_counts = x_train['project_is_approved'].value_counts()
print("Number of projects that are approved for funding    ", x_train_y_value_counts[1]," -> ",round(x_train_y_value_counts[1]/(x_train_y_value_counts[1]+x_train_y_value_counts[0])*100,2),"%")
print("Number of projects that are not approved for funding ",x_train_y_value_counts[0]," -> ",round(x_train_y_value_counts[0]/(x_train_y_value_counts[1]+x_train_y_value_counts[0])*100,2),"%")
print("\n")
# y_value_counts = row1['project_is_approved'].value_counts()
print("X_TEST--------------------------")
x_test_y_value_counts = x_test['project_is_approved'].value_counts()
print("Number of projects that are approved for funding    ", x_test_y_value_counts[1]," -> ",round(x_test_y_value_counts[1]/(x_test_y_value_counts[1]+x_test_y_value_counts[0])*100,2),"%")
print("Number of projects that are not approved for funding ",x_test_y_value_counts[0]," -> ",round(x_test_y_value_counts[0]/(x_test_y_value_counts[1]+x_test_y_value_counts[0])*100,2),"%")
print("\n")

print("X_Val--------------------------")
x_val_y_value_counts = x_val['project_is_approved'].value_counts()
print("Number of projects that are approved for funding    ", x_val_y_value_counts[1]," -> ",round(x_val_y_value_counts[1]/(x_val_y_value_counts[1]+x_val_y_value_counts[0])*100,2),"%")
print("Number of projects that are not approved for funding ",x_val_y_value_counts[0]," -> ",round(x_val_y_value_counts[0]/(x_val_y_value_counts[1]+x_val_y_value_counts[0])*100,2),"%")
print("\n")


# Vectorizing project subject categories
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub = CountVectorizer(lowercase=False, binary=True)

vectorizer_sub.fit(x_train['project_subject_categories'].values)

x_train_project_subject_categories_one_hot = vectorizer_sub.transform(x_train['project_subject_categories'].values)

x_test_project_subject_categories_one_hot  = vectorizer_sub.transform(x_test['project_subject_categories'].values)

x_val_project_subject_categories_one_hot  = vectorizer_sub.transform(x_val['project_subject_categories'].values)
print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_subject_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_subject_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_categories_one_hot.shape)

# Saving model

pickle.dump(vectorizer_sub, open("/email/dsml/models/vectorizer_subject_Category.pkl", "wb"))



# Vectorizing project project_resource_summary categories
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_resource_summary = CountVectorizer(lowercase=False, binary=True)

vectorizer_resource_summary.fit(x_train['project_resource_summary'].values)

x_train_project_resource_summary_one_hot = vectorizer_resource_summary.transform(x_train['project_resource_summary'].values)

x_test_project_resource_summary_one_hot  = vectorizer_resource_summary.transform(x_test['project_resource_summary'].values)

x_val_project_resource_summary_one_hot  = vectorizer_resource_summary.transform(x_val['project_resource_summary'].values)


# x_train['project_subject_categories_encoded'] = x_train_categories_one_hot.toarray()

# x_test['project_subject_categories_encoded']  = x_test_categories_one_hot.toarray()
# print(vectorizer_sub.get_feature_names_out())

print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_resource_summary_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_resource_summary_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_resource_summary_one_hot.shape)

# Saving model

pickle.dump(vectorizer_resource_summary, open("/email/dsml/models/vectorizer_resource_summary.pkl", "wb"))


# Vectorizing project subject sub-categories
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub_sub_category = CountVectorizer(lowercase=False, binary=True)

vectorizer_sub_sub_category.fit(x_train['project_subject_subcategories'].values)

x_train_project_subject_subcategories_one_hot = vectorizer_sub_sub_category.transform(x_train['project_subject_subcategories'].values)

x_test_project_subject_subcategories_one_hot  = vectorizer_sub_sub_category.transform(x_test['project_subject_subcategories'].values)

x_val_project_subject_subcategories_one_hot  = vectorizer_sub_sub_category.transform(x_val['project_subject_subcategories'].values)


print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_subject_subcategories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_subject_subcategories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_subcategories_one_hot.shape)

# Saving model
pickle.dump(vectorizer_sub_sub_category, open("/email/dsml/models/vectorizer_subject_subcategory.pkl", "wb"))


# Vectorizing teacher_prefix
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_teacher_prefix = CountVectorizer(lowercase=False, binary=True)

vectorizer_teacher_prefix.fit(x_train['teacher_prefix'].values)

x_train_teacher_prefix_one_hot = vectorizer_teacher_prefix.transform(x_train['teacher_prefix'].values)

x_test_teacher_prefix_one_hot  = vectorizer_teacher_prefix.transform(x_test['teacher_prefix'].values)

x_val_teacher_prefix_one_hot  = vectorizer_teacher_prefix.transform(x_val['teacher_prefix'].values)


print("Shape of matrix after one hot encoding -> categories: x_train: ",x_test_teacher_prefix_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_teacher_prefix_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_teacher_prefix_one_hot.shape)

# Saving model
pickle.dump(vectorizer_teacher_prefix, open("/email/dsml/models/vectorizer_teacher_prefix.pkl", "wb"))

# Vectorizing school_state
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_school_state = CountVectorizer(lowercase=False, binary=True)

vectorizer_school_state.fit(x_train['school_state'].values)

x_train_school_state_one_hot = vectorizer_school_state.transform(x_train['school_state'].values)

x_test_school_state_one_hot  = vectorizer_school_state.transform(x_test['school_state'].values)

x_val_school_state_one_hot  = vectorizer_school_state.transform(x_val['school_state'].values)

print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_school_state_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_school_state_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_school_state_one_hot.shape)

# Saving model
pickle.dump(vectorizer_school_state, open("/email/dsml/models/vectorizer_school_state.pkl", "wb"))


# Vectorizing project_grade_category
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_grade_category = CountVectorizer(lowercase=False, binary=True)

vectorizer_project_grade_category.fit(x_train['project_grade_category'].values)

x_train_project_grade_category_one_hot = vectorizer_project_grade_category.transform(x_train['project_grade_category'].values)

x_test_project_grade_category_one_hot  = vectorizer_project_grade_category.transform(x_test['project_grade_category'].values)

x_val_project_grade_category_one_hot  = vectorizer_project_grade_category.transform(x_val['project_grade_category'].values)
print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_grade_category_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_grade_category_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_grade_category_one_hot.shape)

# Saving model
pickle.dump(vectorizer_project_grade_category, open("/email/dsml/models/vectorizer_project_grade_category.pkl", "wb"))


# Applying CountVectorizer on project_title

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer_project_title_tfidf = TfidfVectorizer(min_df=10)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_title_category = CountVectorizer(min_df=10,lowercase=False, binary=True)

vectorizer_project_title_category.fit(x_train['project_title'])
x_train_project_titles_tfidf = vectorizer_project_title_category.transform(x_train['project_title'])
x_val_project_titles_tfidf    = vectorizer_project_title_category.transform(x_val['project_title'])
x_test_project_titles_tfidf  = vectorizer_project_title_category.transform(x_test['project_title'])

print("Shape of matrix after TF-IDF -> Title: x_train: ",x_train_project_titles_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_val  : ",x_val_project_titles_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_project_titles_tfidf.shape)

# Saving model
pickle.dump(vectorizer_project_title_category, open("/email/dsml/models/vectorizer_project_title_category.pkl", "wb"))


# Applying CountVectorizer on essay
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_essay_category = CountVectorizer(min_df=10,lowercase=False, binary=True)
vectorizer_project_essay_category.fit(x_train['essay'])

x_train_essay_tfidf = vectorizer_project_essay_category.transform(x_train['essay'])
x_val_essay_tfidf = vectorizer_project_essay_category.transform(x_val['essay'])

x_test_essay_tfidf  = vectorizer_project_essay_category.transform(x_test['essay'])

print("Shape of matrix after TF-IDF -> Title: x_train: ",x_train_essay_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_val: ",x_val_essay_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_essay_tfidf.shape)
# Saving model
pickle.dump(vectorizer_project_essay_category, open("/email/dsml/models/vectorizer_project_essay_category.pkl", "wb"))


# df=pd.DataFrame()
# applying StandardScaler to specific columns
from sklearn.preprocessing import StandardScaler
scaler_transform_numerical_value = StandardScaler()
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# scaler_transform_numerical_value.fit(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
# df[["x","y","z"]]=x_train_teacher_number_of_previously_posted_projects_scaler=scaler_transform_numerical_value.transform(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])


normalizer.fit(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
# df[["x","y","z"]]=x_train_teacher_number_of_previously_posted_projects_scaler=normalizer.transform(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])

x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
x_val[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_val[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
# Saving model
pickle.dump(normalizer, open("/email/dsml/models/normalizer.pkl", "wb"))


from scipy.sparse import hstack
# merging train values

x_train_onehot = hstack((x_train_project_subject_categories_one_hot,
                         x_train_project_resource_summary_one_hot,
                         x_train_project_subject_subcategories_one_hot   ,
                         x_train_teacher_prefix_one_hot    ,
                         x_train_school_state_one_hot  ,
                         x_train_project_grade_category_one_hot  ,
                         x_train_project_titles_tfidf  ,
                         x_train_essay_tfidf,
                         x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]))


print(x_train_onehot.shape)

# merging test value
x_test_onehot = hstack((x_test_project_subject_categories_one_hot,
                         x_test_project_resource_summary_one_hot,
                         x_test_project_subject_subcategories_one_hot   ,
                         x_test_teacher_prefix_one_hot    ,
                         x_test_school_state_one_hot  ,
                         x_test_project_grade_category_one_hot  ,
                         x_test_project_titles_tfidf  ,
                         x_test_essay_tfidf,
                         x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]))

# merging val value
x_val_onehot = hstack((x_val_project_subject_categories_one_hot,
                         x_val_project_resource_summary_one_hot,
                         x_val_project_subject_subcategories_one_hot   ,
                         x_val_teacher_prefix_one_hot    ,
                         x_val_school_state_one_hot  ,
                         x_val_project_grade_category_one_hot  ,
                         x_val_project_titles_tfidf  ,
                         x_val_essay_tfidf,
                         x_val[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]))

print(x_val_onehot.shape)



print()
print("Started training")


# # Compute class weights for imbalanced data
# class_weights = dict(zip(np.unique(y_train), 1 / np.bincount(y_train)))

# # Create the Naive Bayes classifier with class weights
# classifier = GaussianNB(priors=list(class_weights.values()))

# # Train the classifier on the imbalanced training data
# classifier.fit(x_train_onehot, y_train)


classifier = MultinomialNB(class_prior=[0.5, 0.5],alpha= 0.5)
classifier.fit(x_train_onehot, y_train)



# Predictiion on test data
pred_xgboost=classifier.predict(x_test_onehot)
pred_xgboost_train = classifier.predict(x_train_onehot)


#Checking different metrics for bagging model with default hyper parameters
print('Checking different metrics for bagging model with default hyper parameters:\n')
print("Training accuracy: ",classifier.score(x_train_onehot,y_train))
acc_score = accuracy_score(y_test, pred_xgboost)
print('Testing accuracy: ',acc_score)
conf_mat = confusion_matrix(y_test, pred_xgboost)
print('Confusion Matrix: \n',conf_mat)
roc_auc = roc_auc_score(y_test,pred_xgboost)
print('ROC AUC score: ',roc_auc)
class_rep_xgboost= classification_report(y_test,pred_xgboost)
print('Classification Report: \n',class_rep_xgboost)




# mnb_bow = MultinomialNB(class_prior=[0.5, 0.5])
mnb_bow = MultinomialNB(class_prior=[0.8, 0.2])

parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.5,0.6,0.7,0.8,0.9, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
clf = GridSearchCV(mnb_bow, parameters, cv= 10, scoring='roc_auc',verbose=1,return_train_score=True)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestAlpha_1=clf.best_params_['alpha']
bestScore_1=clf.best_score_
print("BEST ALPHA: ",clf.best_params_['alpha']," BEST SCORE: ",clf.best_score_) 

mnb_bow_testModel = MultinomialNB(alpha = bestAlpha_1,class_prior=[0.5, 0.5])
mnb_bow_testModel.fit(x_train_onehot, y_train)

# Saving model
# pickle.dump(mnb_bow_testModel, open("/email/dsml/models/mnb_bow_testModel.pkl", "wb"))
y_test_pred=mnb_bow_testModel.predict_proba(x_test_onehot)[:,1]
print(y_test_pred)

roc_auc = roc_auc_score(y_test,y_test_pred)
print('ROC AUC score: ',roc_auc)


print()
y_val_pred=mnb_bow_testModel.predict_proba(x_val_onehot)[:,1]
# print(y_test_pred)

roc_auc = roc_auc_score(y_val,y_val_pred)
print('ROC AUC score: ',roc_auc)


# Trying Technique

from imblearn.over_sampling import SMOTE
from collections import Counter

smt = SMOTE(k_neighbors=6)
X_sm, y_sm = smt.fit_resample(x_train_onehot, y_train)


print('Resampled dataset shape {}'.format(Counter(y_sm)))

print()
print("Results after using smote technique")
# mnb_bow = MultinomialNB(class_prior=[0.5, 0.5])
mnb_bow = MultinomialNB(class_prior=[0.8, 0.2])

parameters = {'alpha':[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.5,0.6,0.7,0.8,0.9, 1, 5, 10, 50, 100, 500, 1000, 2500, 5000, 10000]}
clf = GridSearchCV(mnb_bow, parameters, cv= 10, scoring='roc_auc',verbose=1,return_train_score=True)
# clf.fit(x_cv_onehot_bow, y_cv)
clf.fit(x_train_onehot,y_train)
train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score']
cv_auc_std= clf.cv_results_['std_test_score']
bestAlpha_1=clf.best_params_['alpha']
bestScore_1=clf.best_score_
print("BEST ALPHA: ",clf.best_params_['alpha']," BEST SCORE: ",clf.best_score_) 

mnb_bow_testModel = MultinomialNB(alpha = bestAlpha_1,class_prior=[0.5, 0.5])
mnb_bow_testModel.fit(X_sm, y_sm)

# Saving model
# pickle.dump(mnb_bow_testModel, open("/email/dsml/models/mnb_bow_testModel.pkl", "wb"))
y_test_pred=mnb_bow_testModel.predict_proba(x_test_onehot)[:,1]
print(y_test_pred)

roc_auc = roc_auc_score(y_test,y_test_pred)
print('ROC AUC score: ',roc_auc)


print()
y_val_pred=mnb_bow_testModel.predict_proba(x_val_onehot)[:,1]
# print(y_test_pred)

roc_auc = roc_auc_score(y_val,y_val_pred)
# print('ROC AUC score: ',roc_auc)