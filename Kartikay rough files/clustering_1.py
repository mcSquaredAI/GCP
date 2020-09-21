

cpdf = pd.read_csv( 'country_lvl_agregate_10features_.csv', header = 0 ).iloc[:,2:]
cpdf.set_index( 'country', inplace=True )

cpdf_o = pd.read_csv( 'country_lvl_agregate_10features_.csv', header = 0 ).iloc[:,2:]
cpdf_o.set_index( 'country', inplace=True )

from sklearn import preprocessing

print(cpdf.columns)

cpdf = cpdf.loc[:, [ 'Populatio', 'mkt_size', 'Market share bio', 'country_lvl_percent_growth', 'No. of products', 'HHI']]


print( np.corrcoef( cpdf.T.values ) )

'''
       [[ 1.       ,  0.03381764, -0.10821651,  0.88721687, -0.22104773,0.32734083],
       [ 0.03381764,  1.        ,  0.37601676, -0.05580201,  0.4929616 ,-0.46205635],
       [-0.10821651,  0.37601676,  1.        ,  0.05090936,  0.70900561,-0.54107989],
       [ 0.88721687, -0.05580201,  0.05090936,  1.        , -0.05992662,0.15299881],
       [-0.22104773,  0.4929616 ,  0.70900561, -0.05992662,  1.        ,-0.7643113 ],
       [ 0.32734083, -0.46205635, -0.54107989,  0.15299881, -0.7643113 ,1.        ]])
       
       High Corr:
       popu and percent_growth
       No. of products and HHI
'''


x = cpdf.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
cpdf = pd.DataFrame(x_scaled , index = cpdf.index)


# cpdf_2015 = copy.deepcopy( cpdf  )
cpdf_2015 = copy.deepcopy( cpdf )

# cpdf_2015.to_csv( 'combined_features_2015onwards.csv' )


cpdf_2015

cpdf_2015_2 = copy.deepcopy(  cpdf_2015.iloc[:,:-1] )

# cpdf_2015_2.max()
# 
# cpdf_2015_2_nm = cpdf_2015_2/cpdf_2015_2.max()

import sklearn 
from sklearn import metrics

# metrics.pairwise.euclidean_distances?





# dm0 = pd.read_csv(fl , header=0 , sep = ' ')

    
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform as sqf




# Dsqf = sqf(dm0.values )
# print ( sdm.values - sdm.values.transpose()) 



distance_cut =  0.50
method = 'ward'

sdm = metrics.pairwise.euclidean_distances( cpdf_2015_2.values )
Dsqf = sqf(sdm, checks = False)
linkage_matrix = linkage(Dsqf, method)

from scipy.cluster.hierarchy import fcluster
max_d = distance_cut
clusters = fcluster(linkage_matrix, max_d, criterion='distance')
clusters

cluster_methodology = 'HC_' + method + '_D=' + str(distance_cut)
cpdf_o[cluster_methodology] = clusters


# printing clusters
labels = cpdf_2015_2.index

max(clusters)
min(clusters)

dic_cluster = {}

for i in range( min(clusters), max(clusters)+1):
    dic_cluster[i] = []
    for inx,v in enumerate(clusters):
        if v == i:
            dic_cluster[i].append( labels[inx] )
            
            
for k, v in dic_cluster.items():
    print('Cluster {}'.format(k))
    print(v)         


# plottting cluster
figure = plt.figure(figsize=(7.5, 5))
den = dendrogram(
    linkage_matrix,
    color_threshold= distance_cut,
    labels = cpdf_2015_2.index
)
plt.title('Hierarchical Clustering Dendrogram (Single)')
plt.xlabel('Currency Symbol')
plt.ylabel('AC measure')
plt.tight_layout()
plt.show()















from sklearn.cluster import KMeans
import numpy as np

for rs in range(6):
    kmeans = KMeans(n_clusters=8, random_state=rs).fit(cpdf_2015_2.values)
    kmeans.labels_
    
    # kmeans.predict([[0, 0], [12, 3]])
    
    kmeans.cluster_centers_
    
    
    clusters =  kmeans.labels_
    
    cluster_methodology = 'Kmeans_'  + 'initialisation=' + str(rs)
    cpdf_o[cluster_methodology] = clusters



cpdf_o.to_csv( 'country_lvl_agregate_10features_based_clusters.csv' )

labels = cpdf_2015_2.index

max(clusters)
min(clusters)

dic_cluster = {}

for i in range( min(clusters), max(clusters)+1):
    dic_cluster[i] = []
    for inx,v in enumerate(clusters):
        if v == i:
            dic_cluster[i].append( labels[inx] )
            
            
for k, v in dic_cluster.items():
    print('Cluster {}'.format(k))
    print(v)         























rt = get_cluster_classes(den)

for k, v in rt.items():
    print('Cluster {}'.format(k))
    print(v)



[den['color_list'],den['ivl']]

pd.DataFrame( den['color_list'],den['ivl'] )

cpdf_2015 = copy.deepcopy( cpdf[ cpdf.index >= '2020-01-01' ] )

# # cpdf_2015.to_csv( 'combined_features_2015onwards.csv' )


cpdf_2015

cpdf_2015_2 = copy.deepcopy(  cpdf_2015.iloc[:,:-1].T )

cpdf_2015_2.max()

cpdf_2015_2_nm = cpdf_2015_2/cpdf_2015_2.max()

import sklearn 

# sklearn.metrics.pairwise.euclidean_distances?


sdm = sklearn.metrics.pairwise.euclidean_distances( cpdf_2015_2_nm.values )


# dm0 = pd.read_csv(fl , header=0 , sep = ' ')

    
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform as sqf



Dsqf = sqf(sdm, checks = False)
# Dsqf = sqf(dm0.values )
# print ( sdm.values - sdm.values.transpose()) 

sdm
linkage_matrix = linkage(Dsqf, 'ward')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix,
    color_threshold=0,
    labels = cpdf_2015_2.index
)
plt.title('Hierarchical Clustering Dendrogram (Single)')
plt.xlabel('Currency Symbol')
plt.ylabel('AC measure')
plt.tight_layout()
plt.show()





