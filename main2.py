import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Load data
data = pd.read_excel('datatraincrimeanaliz.xlsx', engine='openpyxl')

# Print available columns
print("Available columns in the dataset:")
print(data.columns.tolist())

# Properties and target variable
properties = ['Latitude', 'Longitude', 'Time_Occurred', 'Victim_Age']

target = 'Crime_Category'

# Delete missing data
X = data[properties].fillna(data[properties].mean())
y = data[target].fillna(data[target].mode()[0])

# Create custom decision tree (simplified)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X, y)

# Tree visualize and save as PDF
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=properties,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("simplified_decision_tree", format='pdf')

