import urllib.request
import os
import pandas as pd
df_labels = pd.read_csv('./pheno/Phenotypic_V1_0b_preprocessed1.csv')

os.mkdir("dl")

site = ['NYU','OHSU','USM','UCLA_1','UCLA_2']
for row in df_labels.iterrows():
    site_id = row[1]['SITE_ID']
    file_id = row[1]['FILE_ID']
    if file_id == 'no_filename' or site_id not in site:
        continue
    
    url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/"+file_id+"_rois_cc200.1D"
    file_id+="_rois_cc200.1D"
    print(file_id)
    urllib.request.urlretrieve(url,"./dl/"+file_id)

