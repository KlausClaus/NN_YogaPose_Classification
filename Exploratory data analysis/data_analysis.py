import pandas as pd
import plotly.express as px



data1 = pd.read_csv("../Dataset/Yoga-82/yoga_train.txt", header=None, names=['Image_Address', 'Label_Class_6', 'Label_Class_20', 'Label_Class_82'])
data2 = pd.read_csv("../Dataset/Yoga-82/yoga_test.txt", header=None, names=['Image_Address', 'Label_Class_6', 'Label_Class_20', 'Label_Class_82'])
concat_df = pd.concat([data1, data2], axis=0)

print("the first few rows of the data:")
print(concat_df.head())

print("\ndata information:")
print(concat_df.info())

# # sunburst chart to visualize the hierarchical structure
# fig = px.sunburst(concat_df, 
#                   path=['Label_Class_6', 'Label_Class_20', 'Label_Class_82'], 
#                   #values='Count',
#                   title='Hierarchical breakdown by class labels')
# fig.show()

print(concat_df)