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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

df=pd.DataFrame()
df=x_test[['teacher_prefix', 'school_state','project_resource_summary', 'project_title',
       'teacher_number_of_previously_posted_projects',
       'project_subject_categories', 'project_subject_subcategories',
       'project_grade_category', 'essay', 'price', 'quantity']]

df["project_is_approved"]=y_test
df.to_csv("testingData.csv",index=False)

print(df)



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
vectorizer_sub = pickle.load(open("/email/dsml/models/vectorizer_subject_Category.pkl", "rb"))

x_train_project_subject_categories_one_hot = vectorizer_sub.transform(x_train['project_subject_categories'].values)

x_test_project_subject_categories_one_hot  = vectorizer_sub.transform(x_test['project_subject_categories'].values)

x_val_project_subject_categories_one_hot  = vectorizer_sub.transform(x_val['project_subject_categories'].values)
print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_subject_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_subject_categories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_categories_one_hot.shape)


# Vectorizing project project_resource_summary categories
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_resource_summary = pickle.load( open("/email/dsml/models/vectorizer_resource_summary.pkl", "rb"))


x_train_project_resource_summary_one_hot = vectorizer_resource_summary.transform(x_train['project_resource_summary'].values)

x_test_project_resource_summary_one_hot  = vectorizer_resource_summary.transform(x_test['project_resource_summary'].values)

x_val_project_resource_summary_one_hot  = vectorizer_resource_summary.transform(x_val['project_resource_summary'].values)


print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_resource_summary_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_resource_summary_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_resource_summary_one_hot.shape)







# Vectorizing project subject sub-categories
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub_sub_category = pickle.load(open("/email/dsml/models/vectorizer_subject_subcategory.pkl", "rb"))


x_train_project_subject_subcategories_one_hot = vectorizer_sub_sub_category.transform(x_train['project_subject_subcategories'].values)

x_test_project_subject_subcategories_one_hot  = vectorizer_sub_sub_category.transform(x_test['project_subject_subcategories'].values)

x_val_project_subject_subcategories_one_hot  = vectorizer_sub_sub_category.transform(x_val['project_subject_subcategories'].values)


print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_subject_subcategories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_subject_subcategories_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_subject_subcategories_one_hot.shape)


# Vectorizing teacher_prefix
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_teacher_prefix = pickle.load(open("/email/dsml/models/vectorizer_teacher_prefix.pkl", "rb"))


x_train_teacher_prefix_one_hot = vectorizer_teacher_prefix.transform(x_train['teacher_prefix'].values)

x_test_teacher_prefix_one_hot  = vectorizer_teacher_prefix.transform(x_test['teacher_prefix'].values)

x_val_teacher_prefix_one_hot  = vectorizer_teacher_prefix.transform(x_val['teacher_prefix'].values)


print("Shape of matrix after one hot encoding -> categories: x_train: ",x_test_teacher_prefix_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_teacher_prefix_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_teacher_prefix_one_hot.shape)


# Vectorizing school_state
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_school_state = pickle.load(open("/email/dsml/models/vectorizer_school_state.pkl", "rb"))

vectorizer_school_state.fit(x_train['school_state'].values)

x_train_school_state_one_hot = vectorizer_school_state.transform(x_train['school_state'].values)

x_test_school_state_one_hot  = vectorizer_school_state.transform(x_test['school_state'].values)

x_val_school_state_one_hot  = vectorizer_school_state.transform(x_val['school_state'].values)

print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_school_state_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_school_state_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_school_state_one_hot.shape)



# Vectorizing project_grade_category
# we use count vectorizer to convert the values into one
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_grade_category = pickle.load(open("/email/dsml/models/vectorizer_project_grade_category.pkl", "rb"))


x_train_project_grade_category_one_hot = vectorizer_project_grade_category.transform(x_train['project_grade_category'].values)

x_test_project_grade_category_one_hot  = vectorizer_project_grade_category.transform(x_test['project_grade_category'].values)

x_val_project_grade_category_one_hot  = vectorizer_project_grade_category.transform(x_val['project_grade_category'].values)
print("Shape of matrix after one hot encoding -> categories: x_train: ",x_train_project_grade_category_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_val   : ",x_val_project_grade_category_one_hot.shape)
print("Shape of matrix after one hot encoding -> categories: x_test : ",x_test_project_grade_category_one_hot.shape)





# Applying CountVectorizer on project_title

# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer_project_title_tfidf = TfidfVectorizer(min_df=10)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_title_category = pickle.load(open("/email/dsml/models/vectorizer_project_title_category.pkl", "rb"))

x_train_project_titles_tfidf = vectorizer_project_title_category.transform(x_train['project_title'])
x_val_project_titles_tfidf    = vectorizer_project_title_category.transform(x_val['project_title'])
x_test_project_titles_tfidf  = vectorizer_project_title_category.transform(x_test['project_title'])

print("Shape of matrix after TF-IDF -> Title: x_train: ",x_train_project_titles_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_val  : ",x_val_project_titles_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_project_titles_tfidf.shape)



# Applying CountVectorizer on essay
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_project_essay_category = pickle.load(open("/email/dsml/models/vectorizer_project_essay_category.pkl", "rb"))


x_train_essay_tfidf = vectorizer_project_essay_category.transform(x_train['essay'])
x_val_essay_tfidf = vectorizer_project_essay_category.transform(x_val['essay'])

x_test_essay_tfidf  = vectorizer_project_essay_category.transform(x_test['essay'])

print("Shape of matrix after TF-IDF -> Title: x_train: ",x_train_essay_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_val: ",x_val_essay_tfidf.shape)
print("Shape of matrix after TF-IDF -> Title: x_test : ",x_test_essay_tfidf.shape)


df=pd.DataFrame()
# applying StandardScaler to specific columns
from sklearn.preprocessing import StandardScaler
scaler_transform_numerical_value = StandardScaler()
from sklearn.preprocessing import Normalizer
normalizer = pickle.load(open("/email/dsml/models/normalizer.pkl", "rb"))

df[["x","y","z"]]=x_train_teacher_number_of_previously_posted_projects_scaler=normalizer.transform(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])

x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_train[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_test[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])
x_val[['teacher_number_of_previously_posted_projects', 'price',  'quantity']]=normalizer.transform(x_val[['teacher_number_of_previously_posted_projects', 'price',  'quantity']])



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




print()

mnb_bow_testModel = pickle.load(open("/email/dsml/mnb_bow_testModel.pkl", "rb"))


print("Test Score")

pred_xgboost=mnb_bow_testModel.predict(x_test_onehot)
pred_xgboost_train = mnb_bow_testModel.predict(x_train_onehot)
y_test_pred=mnb_bow_testModel.predict_proba(x_test_onehot)[:,1]

#Checking different metrics for bagging model with default hyper parameters
print('Checking different metrics for bagging model with default hyper parameters:\n')
print("Training accuracy: ",mnb_bow_testModel.score(x_train_onehot,y_train))
# acc_score = accuracy_score(y_test_pred, pred_xgboost)
# print('Testing accuracy: ',acc_score)
print('Testing accuracy: ',mnb_bow_testModel.score(x_test_onehot,y_test))


y_test_pred_binary=[]
for i in y_test_pred:
       if i>0.5:
              y_test_pred_binary.append(1)
       else:
              y_test_pred_binary.append(0)

conf_mat = confusion_matrix(y_test_pred_binary, y_test)
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
plt.savefig('/email/dsml/models/TestConfusion.png', bbox_inches='tight')

def plot_roc_curve(true_y, y_prob,name):
    """
    plots the roc curve based of the probabilities
    """
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(fpr, tpr)
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--')

    
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'/email/dsml/models/{name}.png', bbox_inches='tight')



print('Confusion Matrix: \n',conf_mat)
roc_auc = roc_auc_score(y_test,y_test_pred)

# Plot ROC AUC curve
plot_roc_curve(y_test_pred_binary, y_test,"testRoc")


print('ROC AUC score: ',roc_auc)

class_rep_xgboost= classification_report(y_test,y_test_pred_binary)
print('Classification Report: \n',class_rep_xgboost)


print("--------------------------------------------------------------------------------------------------------------------------------------")

print("Validation Score")

y_val_pred=mnb_bow_testModel.predict_proba(x_val_onehot)[:,1]

print('Checking different metrics for bagging model with default hyper parameters:\n')
print("Training accuracy: ",mnb_bow_testModel.score(x_train_onehot,y_train))

print('Validation accuracy: ',mnb_bow_testModel.score(x_val_onehot,y_val))
y_val_pred_binary=[]
for i in y_val_pred:
       if i>0.5:
              y_val_pred_binary.append(1)
       else:
              y_val_pred_binary.append(0)

conf_mat = confusion_matrix(y_val_pred_binary, pred_xgboost)
print('Confusion Matrix: \n',conf_mat)
# plotting confusion matrix
disp = ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
plt.savefig('/email/dsml/models/ValConfusion.png', bbox_inches='tight')


# Plot ROC AUC curve
plot_roc_curve(y_val_pred_binary, y_val,"valRoc")
roc_auc = roc_auc_score(y_val,y_val_pred)
print('ROC AUC score: ',roc_auc)

# Classification report
class_rep_xgboost= classification_report(y_val,y_val_pred_binary)
print('Classification Report: \n',class_rep_xgboost)





