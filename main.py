import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sample_dataset = r"C:\Users\dev_admin\Documents\health_dataset\dataset\heart.csv"

df = pd.read_csv(sample_dataset)

X, y = df.drop("target", axis=1), df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=9)

# scaling the INPUT as some model are very sensitive on the input value magnitude
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

### Scale-insensitive
## Random Forest 
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

## Gaussian Naive bay
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

## Gradient boosting
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)

## Knn clf
knn = KNeighborsClassifier()
knn.fit(X_train_scale, y_train) # KNN are only using the scaled input dataset to predict

## SVM clf
svc = SVC()
svc.fit(X_train_scale, y_train)

#### Evaluation 
forest.score(X_test, y_test)
gb_clf.score(X_test, y_test)
nb_clf.score(X_test, y_test)
knn.score(X_test_scale, y_test)
svc.score(X_test_scale, y_test)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# For the metrics, we can consider on the Accurancy, Precise, and Recall
## Accurancy are diff with Precise
## Recalls are too general as it might provide a False positive, which is good but not perfect. 

## Going on to checking on the Metrics checking
from sklearn.metrics import recall_score

y_preds_forest = forest.predict(X_test)
y_preds_gb = gb_clf.predict(X_test)
y_preds_nb = nb_clf.predict(X_test)
y_preds_knn = knn.predict(X_test_scale)
y_preds_svc = svc.predict(X_test_scale)

print(f"recall_score_forest : {recall_score(y_preds_forest, y_test)}")
print(f"recall_score_gb: {recall_score(y_preds_gb, y_test)}")
print(f"recall_score_nb: {recall_score(y_preds_nb, y_test)}")
print(f"recall_score_knn: {recall_score(y_preds_knn, y_test)}")
print(f"recall_score_svc: {recall_score(y_preds_svc, y_test)}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ploting on the ROC plot to have the visual on the result
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


y_probs = forest.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC curve")
plt.show()

roc_auc_score(y_test, y_probs)