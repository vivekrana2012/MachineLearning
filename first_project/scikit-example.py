from sklearn import tree

features=[[300, 2], [450, 2], [200, 8], [150, 9]]
labels=[0,0,1,1]

clf=tree.DecisionTreeClassifier()
clf=clf.fit(features, labels)

print(clf.predict([[250, 3]]))