import pandas as pd

datatrain =pd.read_csv("F:\wine_X_train.csv")
datatrain.head()

datatest=pd.read_csv("F:\wine_X_test.csv")
datatest = datatest.fillna(1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


train_y = datatrain['quality'].map(lambda x: "1" if x > 6.0 else "0")
val_y = datatest['quality']
data_features = ['Unnamed', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
train_X = datatrain[data_features]
val_X = datatest[data_features]

seed = 1

clf1 = ExtraTreesClassifier(n_estimators=103, warm_start=True, random_state=seed)
clf2 = QuadraticDiscriminantAnalysis(reg_param=0, tol=1.0e-8)
clf3 = DecisionTreeClassifier(min_samples_split=2, random_state=seed)


clf = VotingClassifier(estimators=[
    ('dtc', clf1),
    ('qda', clf2),
    ('etc', clf3),
], voting='hard')
fit = clf.fit(train_X, train_y)
preds = clf.predict(val_X)
print("train_score:", fit.score(train_X, train_y))
print("val_score:", fit.score(val_X, preds))
print(preds.astype(int))