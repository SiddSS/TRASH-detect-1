import json
import os
import pandas as pd
dataset_dir='../'
ann_filepath = os.path.join(dataset_dir , 'annotations_1_val.json')
dataset = json.load(open(ann_filepath, 'r'))
print(dataset.keys())

images=dataset['images']
file_names=[]
image_id=[]
for img in images:
    file_names.append(img['file_name'])
    image_id.append(img['id'])

# print(image_id)
# print(file_names)
#
df=pd.DataFrame()
df['file_name']=file_names
df['image_id']=image_id

df.to_csv('annot_1.csv')


# // extract all the batch1 from the json and make a seprate json of the same structure
