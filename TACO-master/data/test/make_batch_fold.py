import json
import os
import pandas as pd
import shutil

df=pd.read_csv('annot_1.csv')
# for i in range(1,16):
#     fld_name='batch_{}'.format(i)
#     os.mkdir(fld_name)
print(df.head())
for file_ in df['file_name']:
    dest_=file_.split('/')[0]
    file_='../'+file_
    print(file_)
    shutil.copy2(file_,dest_ )
    
