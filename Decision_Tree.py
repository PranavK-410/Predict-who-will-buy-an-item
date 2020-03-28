#Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('shopping.csv')

#Encode labelled data
le = LabelEncoder()
data = dataset.apply(le.fit_transform)

#Split dataset into test and train
from sklearn.model_selection import train_test_split

x = data.iloc[:,0:4]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)

#apply Decision Tree Classifier to splitted data
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(x,y)
decision_tree.score(x,y)
predicted = decision_tree.predict(x_test)
print(predicted)

#Displaying the tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(decision_tree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
