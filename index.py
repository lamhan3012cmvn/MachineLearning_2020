# In[1]: Library and data loading
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint
import joblib

# prep
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score


# models
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve, r2_score
from sklearn.model_selection import cross_val_score

# Neural Network
from sklearn.neural_network import MLPClassifier
# from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

# Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Naive bayes
from sklearn.naive_bayes import GaussianNB

# Stacking
from mlxtend.classifier import StackingClassifier

from subprocess import check_output
print(check_output(["ls", "./input"]).decode("utf8"))

train_df = pd.read_csv('./input/survey.csv')


def print_InforData(data):
    print(data.shape)
    print(data.describe())
    print(data.info())

# print_InforData(train_df)


# In[2]:  Data cleaning
# Xóa các feature comments, state, Timestamp
train_df = train_df.drop(['comments'], axis=1)
train_df = train_df.drop(['state'], axis=1)
train_df = train_df.drop(['Timestamp'], axis=1)
train_df = train_df.drop(['Country'], axis=1)

# print_InforData(train_df)


# In[3]:  Cleaning NaN
# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                  'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                  'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                  'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                  'seek_help']
floatFeatures = []

# Clean the NaN's
for feature in train_df:
    if feature in intFeatures:
        train_df[feature] = train_df[feature].fillna(defaultInt)
    elif feature in stringFeatures:
        train_df[feature] = train_df[feature].fillna(defaultString)
    elif feature in floatFeatures:
        train_df[feature] = train_df[feature].fillna(defaultFloat)
    else:
        print('Error: Feature %s not recognized.' % feature)

# print(train_df.head(5))


# In[3]: clean 'Gender'
gender = train_df['Gender'].str.lower()
gender = train_df['Gender'].unique()

# Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal",
            "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender",
             "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake",
              "female ", "cis-female/femme", "female (cis)", "femail"]

for (row, col) in train_df.iterrows():
    if str.lower(col.Gender) in male_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='male', inplace=True)
    if str.lower(col.Gender) in female_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='female', inplace=True)
    if str.lower(col.Gender) in trans_str:
        train_df['Gender'].replace(
            to_replace=col.Gender, value='trans', inplace=True)

# Get rid of bullshit
stk_list = ['A little about you', 'p']
train_df = train_df[~train_df['Gender'].isin(stk_list)]
# print(train_df['Gender'].unique())

# In[4]: complete missing age with mean
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill with media() values < 18 and > 120
s = pd.Series(train_df['Age'])
s[s < 18] = train_df['Age'].median()
train_df['Age'] = s
s = pd.Series(train_df['Age'])
s[s > 120] = train_df['Age'].median()
train_df['Age'] = s
# print(train_df.Age)
# Ranges of Age
train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=[
                               "0-20", "21-30", "31-65", "66-100"], include_lowest=True)

# Replace "NaN" string from defaultString
train_df['self_employed'] = train_df['self_employed'].replace(
    [defaultString], 'No')
# print(train_df['self_employed'].unique())

train_df['work_interfere'] = train_df['work_interfere'].replace(
    [defaultString], 'Don\'t know')
# print(train_df['work_interfere'].unique())


# In[5]: #Encoding data
def encodingData(train_df):
    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] = labelValue
    for key, value in labelDict.items():
        print(key, value)
    return train_df, labelDict


train_df, labelDict = encodingData(train_df)
# print(train_df.head())

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print(missing_data)


# In[6]: correlation matrix
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# treatment correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'treatment')['treatment'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# Distribiution and density by Age
plt.figure(figsize=(12, 8))
sns.distplot(train_df["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")
g = sns.FacetGrid(train_df, col='treatment', size=5)
g = g.map(sns.distplot, "Age")

# Let see how many people has been treated
plt.figure(figsize=(12, 8))
labels = labelDict['label_Gender']
g = sns.countplot(x="treatment", data=train_df)
g.set_xticklabels(labels)

plt.title('Total Distribuition by treated or not')

o = labelDict['label_age_range']

g = sns.factorplot(x="age_range", y="treatment", hue="Gender", data=train_df,
                   kind="bar",  ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)

plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Age')
# replace legend labels

new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()

o = labelDict['label_family_history']
g = sns.factorplot(x="family_history", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Family History')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)

plt.show()


o = labelDict['label_care_options']
g = sns.factorplot(x="care_options", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Care options')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()

o = labelDict['label_benefits']
g = sns.factorplot(x="care_options", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Benefits')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()

o = labelDict['label_work_interfere']
g = sns.factorplot(x="work_interfere", y="treatment", hue="Gender",
                   data=train_df, kind="bar", ci=None, size=5, aspect=2, legend_out=True)
g.set_xticklabels(o)
plt.title('Probability of mental health condition')
plt.ylabel('Probability x 100')
plt.xlabel('Work interfere')

# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

# Positioning the legend
g.fig.subplots_adjust(top=0.9, right=0.8)
plt.show()


# In[7] define X and y
feature_cols = ["Age", "Gender", "self_employed", "family_history", "work_interfere", "no_employees", "remote_work", "tech_company", "benefits", "care_options", "wellness_program", "seek_help",
                "anonymity", "leave", "mental_health_consequence", "phys_health_consequence", "coworkers", "supervisor", "mental_health_interview", "phys_health_interview", "mental_vs_physical", "obs_consequence"]
X = train_df[feature_cols]
y = train_df.treatment

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42)

methodDict = {}
rmseDict = ()

# In[8] trainning model


def plt_modelFeatureImportant(model):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    labels = []
    for f in range(X.shape[1]):
        labels.append(feature_cols[f])
    # Plot the feature importances of the forest
    plt.figure(figsize=(12, 8))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), labels, rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()


# # SGD Classifier
# print("SGDClassifier")
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train)
# sgd_clf_acc = cross_val_score(
#     sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# joblib.dump(sgd_clf, 'saved_var/SGDClassifier')
# joblib.dump(sgd_clf_acc, 'saved_var/SGDClassifier_acc')
sgd_clf = joblib.load('saved_var/SGDClassifier')
sgd_clf_acc = joblib.load('saved_var/SGDClassifier_acc')
print("SGDClassifier: ", sgd_clf_acc)

# RandomForestClassifier
# print("RandomForestClassifier")
# rdf_clf = RandomForestClassifier()
# rdf_clf = rdf_clf.fit(X_train, y_train)
# forest_acc = cross_val_score(
#     rdf_clf, X_train, y_train, cv=3, scoring="accuracy")
# # joblib.dump(rdf_clf, 'saved_var/RandomForestClassifier')
# # joblib.dump(forest_acc, 'saved_var/RandomForestClassifier_acc')
rdf_clf = joblib.load('saved_var/RandomForestClassifier')
rdf_clf_acc = joblib.load('saved_var/RandomForestClassifier_acc')
print("RandomForestClassifier: ", rdf_clf_acc)
plt_modelFeatureImportant(rdf_clf)


# print("Logistic regression")
# logisticRegresstion = LogisticRegression(solver='liblinear', random_state=42)
# logisticRegresstion.fit(X_train, y_train)
# logisticRegresstion_acc = logisticRegresstion.score(X_test, y_test)
# joblib.dump(logisticRegresstion, 'saved_var/LogisticRegresstion')
# joblib.dump(logisticRegresstion_acc, 'saved_var/LogisticRegresstion_acc')
logisticRegresstion = joblib.load('saved_var/logisticRegresstion')
logisticRegresstion_acc = joblib.load('saved_var/LogisticRegresstion_acc')
print("Logistic regression: ", logisticRegresstion_acc)


# poly_clf = svm.SVC(kernel='poly', degree=2, C=10,
#                    decision_function_shape='ovo').fit(X_train, y_train)
# joblib.dump(poly_clf, 'saved_var/Poly_clf')
poly_clf = joblib.load('saved_var/poly_clf')

# Evaluate
# poly_clf_acc = cross_val_score(
#     poly_clf, X_train, y_train, cv=3, scoring="accuracy")
# joblib.dump(poly_clf_acc, 'saved_var/Poly_clf_acc')

poly_clf_acc = joblib.load('saved_var/Poly_clf_acc')
print("SVM Poly: ", poly_clf_acc)

# In[8] Tuning
# In[8.1] Evaluating a Classification Model.


def evalClassModel(model, y_test, y_pred_class, plot=False):

    print('Accuracy:', metrics.accuracy_score(y_test, y_pred_class))

    print('Null accuracy:\n', y_test.value_counts())

    # calculate the percentage of ones
    print('Percentage of ones:', y_test.mean())

    # calculate the percentage of zeros
    print('Percentage of zeros:', 1 - y_test.mean())

    # Comparing the true and predicted response values
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])

    # Conclusion:
    # Classification accuracy is the easiest classification metric to understand
    # But, it does not tell you the underlying distribution of response values
    # And, it does not tell you what "types" of errors your classifier is making

    # Confusion matrix
    # save confusion matrix and slice into four pieces
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    # [row, column]
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # visualize Confusion Matrix
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Metrics computed from a confusion matrix
    # Classification Accuracy: Overall, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)

    # Classification Error: Overall, how often is the classifier incorrect?
    print('Classification Error:', 1 -
          metrics.accuracy_score(y_test, y_pred_class))

    # False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)

    # Precision: When a positive value is predicted, how often is the prediction correct?
    print('Precision:', metrics.precision_score(y_test, y_pred_class))

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    print('AUC Score:', metrics.roc_auc_score(y_test, y_pred_class))

    # calculate cross-validated AUC
    print('Cross-validated AUC:', cross_val_score(model,
                                                  X, y, cv=10, scoring='roc_auc').mean())

    ##########################################
    # Adjusting the classification threshold
    ##########################################
    # print the first 10 predicted responses
    # 1D array (vector) of binary values (0, 1)
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])

    # print the first 10 predicted probabilities of class membership
    print('First 10 predicted probabilities of class members:\n',
          model.predict_proba(X_test)[0:10])

    # print the first 10 predicted probabilities for class 1
    model.predict_proba(X_test)[0:10, 1]

    # store the predicted probabilities for class 1
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    print(y_pred_prob)
    if plot == True:
        # histogram of predicted probabilities
        # adjust the font size
        plt.rcParams['font.size'] = 12
        # 8 bins
        plt.hist(y_pred_prob, bins=8)

        # x-axis limit from 0 to 1
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities')
        plt.xlabel('Predicted probability of treatment')
        plt.ylabel('Frequency')

    # predict treatment if the predicted probability is greater than 0.3
    # it will return 1 for all values above 0.3 and 0 otherwise
    # results are 2D so we slice out the first column
    y_pred_prob = y_pred_prob.reshape(-1, 1)
    y_pred_class = binarize(y_pred_prob, 0.3)[0]

    # print the first 10 predicted probabilities
    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    ##########################################
    # ROC Curves and Area Under the Curve (AUC)
    ##########################################

    # Question: Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?
    # Answer: Plot the ROC curve!

    # AUC is the percentage of the ROC plot that is underneath the curve
    # Higher value = better classifier
    roc_auc = metrics.roc_auc_score(y_test, y_pred_prob)

    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # we pass y_test and y_pred_prob
    # we do not use y_pred_class, because it will give incorrect results without generating an error
    # roc_curve returns 3 objects fpr, tpr, thresholds
    # fpr: false positive rate
    # tpr: true positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()

        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for treatment classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = metrics.confusion_matrix(y_test, predict_mine)
    print(confusion)

    return accuracy

# In[8.2] Tuning with cross validation score
##########################################
# Tuning with cross validation score
##########################################


def tuningCV(knn):

    # search for an optimal value of K for KNN
    k_range = list(range(1, 31))
    k_scores = []
    if 0:
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
            k_scores.append(scores.mean())
        joblib.dump(k_scores, 'saved_var/tuningCV_Knn')
    else:
        k_scores = joblib.load('saved_var/tuningCV_Knn')
    print(k_scores)
    # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

# In[8.4] Tuning with RandomizedSearchCV


def tuningRandomizedSearchCV(model, modelName, param_dist, flag=True):
    best_scores = []
    modelName = 'saved_var/tuningRandomizedSearchCV_' + modelName
    if flag == False:
        rand = RandomizedSearchCV(model, param_dist, cv=5,
                                  scoring='accuracy', n_iter=10, random_state=5)
        rand.fit(X, y)

        for _ in range(20):
            rand = RandomizedSearchCV(
                model, param_dist, cv=5, scoring='accuracy', n_iter=10)
            rand.fit(X, y)
            best_scores.append(round(rand.best_score_, 3))

        joblib.dump(best_scores, modelName)
    else:
        best_scores = joblib.load(modelName)
    print(best_scores)


def Knn(flag=True):
    # Calculating the best parameters
    knn = KNeighborsClassifier(n_neighbors=5)

    tuningCV(knn)
    # define the parameter values that should be searched
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']

    # specify "parameter distributions" rather than a "parameter grid"
    param_dist = dict(n_neighbors=k_range, weights=weight_options)
    tuningRandomizedSearchCV(knn, 'Knn', param_dist)

    # train a KNeighborsClassifier model on the training set
    if flag == False:
        knn = KNeighborsClassifier(n_neighbors=27, weights='uniform')
        knn.fit(X_train, y_train)
        joblib.dump(knn, 'saved_var/knn')
    else:
        knn = joblib.load('saved_var/knn')

    # make class predictions for the testing set
    y_pred_class = knn.predict(X_test)

    print('########### KNeighborsClassifier ###############')

    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['KNN'] = accuracy_score * 100


Knn()


def logisticRegression(flag=False):
    # train a logistic regression model on the training set
    if flag == False:
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        joblib.dump(logreg, 'saved_var/logreg')
    else:
        logreg = joblib.load('saved_var/logreg')

    # make class predictions for the testing set
    y_pred_class = logreg.predict(X_test)

    print('########### Logistic Regression ###############')

    accuracy_score = evalClassModel(logreg, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Log. Regres.'] = accuracy_score * 100


logisticRegression()
# In[9]:
# In[9.1]: Random Forests


def randomForest(flag=False):
    # Calculating the best parameters
    forest = RandomForestClassifier(n_estimators=20)

    featuresSize = feature_cols.__len__()
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, featuresSize),
                  "min_samples_split": randint(2, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    tuningRandomizedSearchCV(forest, 'randomForest', param_dist)

    # Building and fitting my_forest
    if flag == False:
        forest = RandomForestClassifier(
            max_depth=None, min_samples_leaf=8, min_samples_split=2, n_estimators=20, random_state=1)
        my_forest = forest.fit(X_train, y_train)
        joblib.dump(my_forest, 'saved_var/my_forest')
    else:
        my_forest = joblib.load('saved_var/my_forest')
    # make class predictions for the testing set
    y_pred_class = my_forest.predict(X_test)

    print('########### Random Forests ###############')

    accuracy_score = evalClassModel(my_forest, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['R. Forest'] = accuracy_score * 100


randomForest()
# In[9.2] Bagging


def bagging(flag=False):
    if flag == False:
        bag = joblib.load('saved_var/bagging')
    else:
        # Building and fitting
        bag = BaggingClassifier(DecisionTreeClassifier(
        ), max_samples=1.0, max_features=1.0, bootstrap_features=False)
        bag.fit(X_train, y_train)
        joblib.dump(bag, 'saved_var/bagging')
    # make class predictions for the testing set
    y_pred_class = bag.predict(X_test)

    print('########### Bagging ###############')

    accuracy_score = evalClassModel(bag, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Bagging'] = accuracy_score * 100


bagging()
# In[9.3] Boosting


def boosting(flag=False):
    if flag == False:
        boost = joblib.load('saved_var/boosting')
    else:
        # Building and fitting
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        boost = AdaBoostClassifier(base_estimator=clf, n_estimators=500)
        boost.fit(X_train, y_train)
        joblib.dump(boost, 'saved_var/boosting')

    # make class predictions for the testing set
    y_pred_class = boost.predict(X_test)

    print('########### Boosting ###############')

    accuracy_score = evalClassModel(boost, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Boosting'] = accuracy_score * 100


boosting()
# In[9.4] Stacking


def stacking(flag=False):
    if flag == False:
        stack = joblib.load('saved_var/stacking')
    else:
        # Building and fitting
        clf1 = KNeighborsClassifier(n_neighbors=1)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = GaussianNB()
        lr = LogisticRegression()
        stack = StackingClassifier(
            classifiers=[clf1, clf2, clf3], meta_classifier=lr)
        stack.fit(X_train, y_train)
        joblib.dump(stack, 'saved_var/stacking')

    # make class predictions for the testing set
    y_pred_class = stack.predict(X_test)

    print('########### Stacking ###############')

    accuracy_score = evalClassModel(stack, y_test, y_pred_class, True)

    # Data for final graph
    methodDict['Stacking'] = accuracy_score * 100


stacking(True)
# joblib.dump(methodDict, 'saved_var/methodDict')
# print(methodDict)
