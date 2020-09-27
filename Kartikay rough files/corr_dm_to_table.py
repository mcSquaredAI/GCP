
import pandas as pd
import copy
import numpy as np

# pdf_cnt_ind_mrk2 = pd.read_csv( 'per_capita_trend_normalised_mkt_size_plc.csv' )

pdf_cnt_ind_mrk2 = pd.read_csv( 'country_lvl_aligned_ts_max.csv', header = [0,1] , index_col=0)

pdf_cnt_ind_mrk2.dropna?

pdf_cnt_ind_mrk21 = copy.deepcopy( pdf_cnt_ind_mrk2.iloc[:,:2] )

pdf_cnt_ind_mrk3 = pdf_cnt_ind_mrk21[~(pdf_cnt_ind_mrk21 == -0.01).any(axis=1)]




N = len( pdf_cnt_ind_mrk2.columns )

dmc = np.array( [[np.nan]*N]*N )

visual_pdf = []
for i1 in range(N-1):
    # i1 = 0
    print(i1)
    for i2 in range(i1+1 , N):
        # i2 = 5
        pdf_cnt_ind_mrk21 = copy.deepcopy( pdf_cnt_ind_mrk2.iloc[:,[i1,i2]] )
        pdf_cnt_ind_mrk3 = pdf_cnt_ind_mrk21[~(pdf_cnt_ind_mrk21 == -0.01).any(axis=1)]
        # np.corrcoef?
        # np.corrcoef( pdf_cnt_ind_mrk3.iloc[:,0].values , pdf_cnt_ind_mrk3.iloc[:,1].values )
        dmc[i1,i2]  = np.corrcoef(  pdf_cnt_ind_mrk3.T.values )[0,1]
        
        visual_pdf.append( [ dmc[i1,i2] , pdf_cnt_ind_mrk3.shape[0] ] + list( pdf_cnt_ind_mrk2.columns[i1] ) + list( pdf_cnt_ind_mrk2.columns[i2] )   )
        
visual_pdf = pd.DataFrame( visual_pdf )

visual_pdf.columns = ['correlation', 'no. of points' ,'set1_country','set1_ind',  'set2_country','set2_ind']

# visual_pdf.to_csv( 'country_indication_level_plc_corr.csv' )        
   
visual_pdf.to_csv( 'country_level_plc_corr.csv' )  
        
# dmc.fill(  )
np.fill_diagonal(  dmc , 0 )

dmc0 = np.nan_to_num( dmc )
dmc0t = np.transpose( dmc0 )

dmct = np.transpose( dmc )

dmc02 = dmc0 + dmc0t

np.fill_diagonal(  dmc02 , 1 )


dmc03 = 1 - dmc02
# # sdm = sklearn.metrics.pairwise.euclidean_distances( cpdf_2015_2_nm.values )


# dm0 = pd.read_csv(fl , header=0 , sep = ' ')



# dm0 = pd.read_csv(fl , header=0 , sep = ' ')

    
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform as sqf

dmc03_cnt_ind = copy.deepcopy( dmc03 )

Dsqf = sqf( dmc03_cnt_ind , checks = False)
# Dsqf = sqf(dm0.values )
# print ( sdm.values - sdm.values.transpose()) 

# sdm
linkage_matrix = linkage(Dsqf, 'complete')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix,
    color_threshold=0,
    labels = pdf_cnt_ind_mrk2.columns 
)
plt.title('Hierarchical Clustering Dendrogram (Single)')
plt.xlabel('Currency Symbol')
plt.ylabel('AC measure')
plt.tight_layout()
plt.show()





