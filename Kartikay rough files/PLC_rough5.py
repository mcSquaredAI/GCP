#!/usr/bin/env python
# coding: utf-8

# In[638]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.pyplot import figure
from statsmodels.tsa.stattools import adfuller


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D


# In[639]:


data = pd.read_csv('combined_timeseries.csv')


# In[616]:


data = data.drop("Unnamed: 0",axis=1)
data['date']= pd.to_datetime(data['date']) 


# In[617]:


data['date_year'] = pd.DatetimeIndex(data['date']).year
data['date_month'] = pd.DatetimeIndex(data['date']).month
data['date_day'] = pd.DatetimeIndex(data['date']).day


# In[618]:


product_list = data['product'].unique()
product_list


# In[628]:


indication_list = data['indication'].unique()
indication_list


# In[629]:


#Market Share


# In[631]:


def get_market_share_by_ind(ind,product,data):
    product_data = data[data['product']==product]
    indication_ms = data.groupby(['indication']).sum()['value'].to_dict()
    deno = 0
    num = 0
    deno = indication_ms[ind]
    num = product_data.groupby(['indication']).sum()['value'][ind]
    return round(num/deno * 100,4)


# In[632]:


def get_market_share_at_datetime(product,data,datetime):
    date_data = data[data['date']==datetime]
    indication_ms = date_data.groupby(['indication']).sum()['value'].to_dict()
    product_data = date_data[date_data['product']==product]
    deno = 0
    num = 0
    for ind in list(dict.fromkeys(date_data['indication'])):
        deno += indication_ms[ind]
    num = product_data['value'].sum()
    return round(num/deno * 100,4)


# In[633]:


def get_market_share_at_datetime_by_indication(product,indication,data,datetime):
    date_data = data[data['date']==datetime]
    indication_data = date_data[date_data['indication']==indication]
    product_list_ind = indication_data['product'].unique()
    product_data = indication_data[indication_data['product']==product]
    deno = 0
    num = 0
    deno = indication_data['value'].sum()
    num = product_data['value'].sum()
    return round(num/deno * 100,4)


# In[634]:


def get_market_share_at_datetime_by_indication_for_country(product,indication,country,data,datetime):
    date_data = data[data['date']==datetime][data['country']==country]
    indication_data = date_data[date_data['indication']==indication]
    product_list_ind = indication_data['product'].unique()
    product_data = indication_data[indication_data['product']==product]
    deno = 0
    num = 0
    deno = indication_data['value'].sum()
    num = product_data['value'].sum()
    return round(num/deno * 100,4)


# In[636]:


product_ms_ind = {}
for ind in list(dict.fromkeys(data['indication'])):
    product_ms_ind[ind] = {}
    for prod in list(dict.fromkeys(data[data['indication']==ind]['product'])):
        product_ms_ind[ind][prod] = get_market_share_by_ind(ind,prod,data)


# In[637]:


for ind in product_ms_ind.keys():
    print(ind)
    # plt.figure(figsize=(10,10))
    # plt.barh(*zip(*product_ms_ind[ind].items()))
    # plt.show()
    


# In[564]:


#PLC Curves

#country
#indication


# In[565]:


def get_plc_curve_yearly(product):
    product_data = data[data['product']==product]
    # plt.plot(product_data.groupby(data['date_year']).sum()['value'])
    # plt.show()
    return product_data.groupby(data['date_year']).sum()['value']


# In[566]:


def get_plc_curve_monthly(product):
    product_data = data[data['product']==product]
    # plt.plot(product_data.groupby(product_data['date']).sum()['value'])
    # plt.show()
    return product_data.groupby(product_data['date']).sum()['value']

# In[567]:


def get_plc_curve_monthly_by_country(product,country):
    product_country_data = data[data['product']==product][data['country']==country]
    # plt.plot(product_country_data.groupby(product_country_data['date']).sum()['value'])
    # plt.show()
    return product_country_data.groupby(product_country_data['date']).sum()['value']


# In[568]:


def get_plc_curve_monthly_by_indication(product,indication):
    product_country_data = data[data['product']==product][data['indication']==indication]
    plt.plot(product_country_data.groupby(product_country_data['date']).sum()['value'])
    plt.show()


# In[569]:

# import copy
def get_plc_curve_monthly_by_country_and_ind(product,country,indication):
    product_country_data = data[data['product']==product][data['country']==country][data['indication']==indication]
    
    return product_country_data.groupby(product_country_data['date']).sum()['value']
    # 
    # plt.plot(product_country_data.groupby(product_country_data['date']).sum()['value'])
    # 
    # plt.show()


# In[570]:


get_plc_curve_monthly("prod15")


# In[571]:


for ind in indication_list :
    print(ind)
#     print(data[data['indication']==ind].country.nunique()
#     print(data[data['indication']==ind].date.min()
#     print(data[data['indication']==ind].date.max()
    print(data[data['indication']==ind].value.sum())


# In[572]:


data.columns


# In[573]:


#Launch Date


# In[574]:


launch_date_raw = {}


# In[575]:


for prod in product_list :
    launch_date_raw[prod] = data[data['product'] == prod].date.min()


# In[576]:


launch_date_raw


# In[577]:


data[data['bioSimilar']=='prod43']['product'].unique()


# In[578]:


indication_list


# In[579]:


for prod in data['product'].unique() :
    print(prod,data[data['product']==prod]['country'].nunique())


# In[580]:


country_list = data['country'].unique()


# In[581]:


data['country'].nunique()


# In[582]:


PRODUCT = 'prod20'
INDICATION = 'indK'
COUNTRY = 'United Kingdom'
#can you plot biosimilars too - combined biosimilar market share


# In[583]:


#What is PLC of Product for Indication
print("Overall PLC for ",INDICATION)
get_plc_curve_monthly_by_indication(PRODUCT,INDICATION)
#What is the market share for the Indication?
print("Overall Market Share for Indication ",INDICATION," across time and country : ",get_market_share_by_ind(INDICATION,PRODUCT,data))
#Market Share Over Time
market_share_over_time = {}
product_data = data[data['product'] == PRODUCT]
for date in list(dict.fromkeys(sorted(product_data['date']))):
    market_share_over_time[date] = get_market_share_at_datetime_by_indication(PRODUCT,INDICATION,data,date)
print("Market Share over Time Across Country for ",INDICATION)
# # # plt.figure(figsize=(10,10))
# # # plt.plot(*zip(*market_share_over_time.items()))
# # # plt.show()


# In[584]:


#How does it vary by country?
product_ind_data = data.loc[(data['product']==PRODUCT)&(data['indication']==INDICATION)]
for country in product_ind_data['country'].unique():
    print(country)
    get_plc_curve_monthly_by_country_and_ind(PRODUCT,country,INDICATION)

#filter to not print blank graphs 


data.columns



prd_iniq = data['product'].unique()
len(prd_iniq)

cnt_uniq = data['country'].unique()
len(cnt_uniq)

ind_uniq = data['indication'].unique()
len(ind_uniq)

plc_dict = {}

s=-1
for tmp_prd in prd_iniq:
    for tmp_cnt in cnt_uniq:
        for tmp_ind in ind_uniq:
            s = s+1
            if s%100 == 0:
                print(s)
            plc_dict[(tmp_cnt,tmp_prd,tmp_ind)] = get_plc_curve_monthly_by_country_and_ind(tmp_prd,tmp_cnt,tmp_ind)
       

import copy

plc_dict_copy = copy.deepcopy( plc_dict )
plc_dict_copy2 = copy.deepcopy( plc_dict_copy )


for tmp_prd in prd_iniq:
    for tmp_cnt in cnt_uniq:
        for tmp_ind in ind_uniq:
            plc_dict[(tmp_cnt,tmp_prd,tmp_ind)] = pd.DataFrame( plc_dict_copy[(tmp_cnt,tmp_prd,tmp_ind)] )
            plc_dict[(tmp_cnt,tmp_prd,tmp_ind)].columns = [(tmp_cnt,tmp_prd,tmp_ind)]

tmp_pdf =   plc_dict[('United Kingdom', 'prod22', 'indI')]  

s= -1
for ky,vl in plc_dict.items():
    # print(ky)
    # ky = ('Australia', 'prod36', 'indK')
    # vl = plc_dict[('Australia', 'prod36', 'indK')]
    s = s+1
    if s%100 == 0:
        print(s)
        
    if s!=0 and len(vl) > 0:
        # print(vl)
        tmp_pdf = tmp_pdf.join(  vl, how='outer'  )
        
    
tmp_pdf.to_csv('Contry_ind_prdt_plc.csv')

tmp_pdf = pd.read_csv( 'Contry_ind_prdt_plc.csv' , index_col='date')

tmp_pdf.columns

import copy
pdf = copy.deepcopy( tmp_pdf )


pdft = copy.deepcopy( pdf.T )

pdft['country'] =  np.array( [ [i[0],i[1],i[2]] for i in pdft.index ] )[:,0]
pdft['prod'] =  np.array( [ [i[0],i[1],i[2]] for i in pdft.index ] )[:,1]
pdft['ind'] =  np.array( [ [i[0],i[1],i[2]] for i in pdft.index ] )[:,2]


pdft.shape

pdft_grp = pdft.groupby( 'prod' )

pdf_cnt_ind_plc = pd.DataFrame([])

for k,v in pdft_grp:
    print(k)
    # print(v.columns)
    
    vt = v.T[:-3]
    
    vt.sum(axis=1).values
    
    pdf_cnt_ind_plc[  v.index[0][0:4:2]  ] =  vt.sum(axis=1).values
    
    

country_pop_gdp = pd.read_csv( 'country_pop_gdp.csv',encoding='ISO-8859-1' )
    


l1 = [str(i).lower() for i in country_pop_gdp['Country'].values]

country_pop_gdp['country'] =   l1


country_pop_gdp.set_index( 'country', inplace=True )




country_pop_gdp.columns

country_pop_gdp2 = copy.deepcopy( country_pop_gdp[1:] )

country_pop_gdp2['gdp'] = [ int(i[2:-1].replace(',','')) for i in country_pop_gdp.loc[:,[' GDP ']].values.reshape(1,-1)[0][1:] ]

country_pop_gdp2['pop'] =[ int(i.replace(',','')) for i in country_pop_gdp.loc[:,['Population']].values.reshape(1,-1)[0][1:] ] 
# country_pop_gdp.Population
# # 
# # l2 = np.unique( [str(i).lower() for i in pdft['country'].values] )
# # 
# # pdft['country'] = [str(i).lower() for i in pdft['country'].values]
# # 
# # 

country_pop_gdp2.to_csv( 'country_pop_gdp2.csv' )


pdf_cnt_ind_mrk = {}

for tmp_cnt in cnt_uniq:
    # tmp_cnt = 'Argentina'
    for tmp_ind in ind_uniq:
        # tmp_ind = 'indI'
        rcols = [ i for i in pdft.index if tmp_cnt in i and tmp_ind in i ]
        print(rcols)
        
        pdf.loc[:, rcols ].sum(axis=1).values
        
        pdf_cnt_ind_mrk[ (tmp_cnt, tmp_ind) ] = pdf.loc[:, rcols ].sum(axis=1).values
        

pdf_cnt_ind_mrk2 = pd.DataFrame.from_dict( pdf_cnt_ind_mrk )
pdf_cnt_ind_mrk2.index = pdf.index


# pdf_cnt_ind_mrk2.to_csv( 'country_indication_level_plc.csv' )

pdf_cnt_ind_mrk2.dropna?

pdf_cnt_ind_mrk21 = copy.deepcopy( pdf_cnt_ind_mrk2.iloc[:,:2] )

pdf_cnt_ind_mrk3 = pdf_cnt_ind_mrk21[~(pdf_cnt_ind_mrk21 == 0).any(axis=1)]




N = len( pdf_cnt_ind_mrk2.columns )

dmc = np.array( [[np.nan]*N]*N )

visual_pdf = []
for i1 in range(N-1):
    # i1 = 0
    print(i1)
    for i2 in range(i1+1 , N):
        # i2 = 5
        pdf_cnt_ind_mrk21 = copy.deepcopy( pdf_cnt_ind_mrk2.iloc[:,[i1,i2]] )
        pdf_cnt_ind_mrk3 = pdf_cnt_ind_mrk21[~(pdf_cnt_ind_mrk21 == 0).any(axis=1)]
        # np.corrcoef?
        # np.corrcoef( pdf_cnt_ind_mrk3.iloc[:,0].values , pdf_cnt_ind_mrk3.iloc[:,1].values )
        dmc[i1,i2]  = np.corrcoef(  pdf_cnt_ind_mrk3.T.values )[0,1]
        
        visual_pdf.append( [ dmc0[i1,i2] , pdf_cnt_ind_mrk3.shape[0] ] + list( pdf_cnt_ind_mrk2.columns[i1] ) + list( pdf_cnt_ind_mrk2.columns[i2] )   )
        
visual_pdf = pd.DataFrame( visual_pdf )

visual_pdf.columns = ['correlation', 'no. of points' ,'set1_country','set1_ind',  'set2_country','set2_ind']

# visual_pdf.to_csv( 'country_indication_level_plc_corr.csv' )        
        
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












pdf_cnt_mrk = {}

for tmp_cnt in cnt_uniq:
    # tmp_cnt = 'Argentina'
    # tmp_ind = 'indI'
    rcols = [ i for i in pdft.index if tmp_cnt in i ]
    print(rcols)
    
    pdf.loc[:, rcols ].sum(axis=1).values
    
    pdf_cnt_mrk[tmp_cnt] = pdf.loc[:, rcols ].sum(axis=1).values
    
pdf_cnt_mrk2 =  pd.DataFrame.from_dict( pdf_cnt_mrk )

pdf_cnt_mrk2.index = pdf.index





pdf_cnt_mrk2.dropna?

pdf_cnt_mrk21 = copy.deepcopy( pdf_cnt_mrk2.iloc[:,:2] )

pdf_cnt_mrk3 = pdf_cnt_mrk21[~(pdf_cnt_mrk21 == 0).any(axis=1)]


# pdf_cnt_mrk2.to_csv( 'country_level_plc.csv' )

N = len( pdf_cnt_mrk2.columns )

dmc = np.array( [[np.nan]*N]*N )

for i1 in range(N-1):
    # i1 = 0
    print(i1)
    for i2 in range(i1+1 , N):
        # i2 = 5
        pdf_cnt_mrk21 = copy.deepcopy( pdf_cnt_mrk2.iloc[:,[i1,i2]] )
        pdf_cnt_mrk3 = pdf_cnt_mrk21[~(pdf_cnt_mrk21 == 0).any(axis=1)]
        # np.corrcoef?
        # np.corrcoef( pdf_cnt_mrk3.iloc[:,0].values , pdf_cnt_mrk3.iloc[:,1].values )
        dmc[i1,i2]  = np.corrcoef(  pdf_cnt_mrk3.T.values )[0,1]
        
        
        
# dmc.fill(  )
np.fill_diagonal(  dmc , 0 )

dmc0 = np.nan_to_num( dmc )
dmc0t = np.transpose( dmc0 )

# dmct = np.transpose( dmc )

dmc02 = dmc0 + dmc0t

np.fill_diagonal(  dmc02 , 1 )


dmc03 = 1 - dmc02
# # sdm = sklearn.metrics.pairwise.euclidean_distances( cpdf_2015_2_nm.values )

visual_pdf = []

for i1 in range(N-1):
    # i1 = 0
    print(i1)
    for i2 in range(i1+1 , N):
        # i2 = 5
        visual_pdf.append( [ dmc0[i1,i2], pdf_cnt_mrk2.columns[i1], pdf_cnt_mrk2.columns[i2]  ]  )

visual_pdf = pd.DataFrame( visual_pdf )

visual_pdf.columns = ['correlation', 'country1', 'country2']

# visual_pdf.to_csv( 'country_level_plc_corr.csv' )
# dm0 = pd.read_csv(fl , header=0 , sep = ' ')

    
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform as sqf

dmc03_cnt_lvl = copy.deepcopy( dmc03 )

Dsqf = sqf( dmc03_cnt_lvl , checks = False)
# Dsqf = sqf(dm0.values )
# print ( sdm.values - sdm.values.transpose()) 

# sdm
linkage_matrix = linkage(Dsqf, 'complete')
figure = plt.figure(figsize=(7.5, 5))
dendrogram(
    linkage_matrix,
    color_threshold=0,
    labels = pdf_cnt_mrk2.columns 
)
plt.title('Hierarchical Clustering Dendrogram (Single)')
plt.xlabel('Currency Symbol')
plt.ylabel('AC measure')
plt.tight_layout()
plt.show()




[ print(i) for i in np.sort( np.unique( dmc03 ) )]












































pdf_cnt_mrk = {}

for tmp_cnt in cnt_uniq:
    # tmp_cnt = 'Argentina'
    # tmp_ind = 'indI'
    rcols = [ i for i in pdft.index if tmp_cnt in i ]
    print(rcols)
    
    pdf.loc[:, rcols ].sum(axis=1).values
    
    pdf_cnt_mrk[tmp_cnt] = pdf.loc[:, rcols ].sum(axis=1).values
    
pdf_cnt_mrk2 =  pd.DataFrame.from_dict( pdf_cnt_mrk )

pdf_cnt_mrk2.index = pdf.index



cnt_gdp_pop = {}
cnt_gdp_pop_lst = []

for i in pdf_cnt_mrk2.columns:
    il = i.lower()
    cnt_gdp_pop[il] = { 'gdp':country_pop_gdp2.loc[il,['gdp', 'pop']].values[0], 'pop':country_pop_gdp2.loc[il,['gdp', 'pop']].values[1]  }
    cnt_gdp_pop_lst.append(  [il] + country_pop_gdp2.loc[il,['gdp', 'pop']].values.tolist() )
    
    
cnt_gdp_pop_lst = pd.DataFrame( np.array( cnt_gdp_pop_lst ) ).T
cnt_gdp_pop_lst.index = ['cnt','gdp', 'pop']

cnt_gdp_pop_lst.columns = cnt_gdp_pop_lst.loc[['cnt'],:].values[0]

cnt_gdp_pop_lst.values[2,:]

pdf_cnt_mrkperpop = pd.DataFrame( pdf_cnt_mrk2.values/ np.array([ int(i) for i in cnt_gdp_pop_lst.values[2,:] ]) )
pdf_cnt_mrkperpop.index = pdf_cnt_mrk2.index

pdf_cnt_mrkperpop.columns = pdf_cnt_mrk2.columns

cnt_gdp_pop_lstt0 = cnt_gdp_pop_lst.T
cnt_gdp_pop_lstt0['gdp'] = [int(i) for i in cnt_gdp_pop_lstt0['gdp'].values]

cnt_gdp_pop_lstt0['pop'] = [int(i) for i in cnt_gdp_pop_lstt0['pop'].values]


cnt_gdp_pop_lstt0['gdpcapita'] =  cnt_gdp_pop_lstt0['gdp']/cnt_gdp_pop_lstt0['pop']

cnt_gdp_pop_pdf = pd.DataFrame( np.array( [cnt_gdp_pop_lstt0['gdpcapita'].values.tolist()]*185 ) )

cnt_gdp_pop_pdf.index = pdf_cnt_mrk2.index
cnt_gdp_pop_pdf.columns =  pdf_cnt_mrk2.columns


# # # # # # # # # # # # # # # # # 
cnt_gdp_pop_pdf['feature'] = 'per_capita_gdp'
pdf_cnt_mrkperpop['feature'] = 'per_capita_marketsize'
pdf_cnt_mrk2['feature'] = 'market_size'
# # # # # # # # # # # # # # # # # 

# # # # # # # # # # # # # # # # # 
cnt_gdp_pop_pdf
pdf_cnt_mrkperpop
pdf_cnt_mrk2[ pdf_cnt_mrk2.index >= '2015-01-01' ]
# # # # # # # # # # # # # # # # # 



biosimilars = pd.read_csv( 'biosimilars.csv' )

biosimilars.set_index( 'date' , inplace=True )

biosimilars['feature'] = 'no of biosimilars'


biosimilars.fillna(-1, inplace=True)

biosimilars





cpdf = cnt_gdp_pop_pdf.append( pdf_cnt_mrkperpop )

cpdf = copy.deepcopy( cpdf[ cpdf.index >= '2015-01-01' ] )

cpdf = cpdf.append( biosimilars )

# pdf.diff()



















# cpdf_2015 = copy.deepcopy( cpdf  )
cpdf_2015 = copy.deepcopy( cpdf )

# cpdf_2015.to_csv( 'combined_features_2015onwards.csv' )


cpdf_2015

cpdf_2015_2 = copy.deepcopy(  cpdf_2015.iloc[:,:-1].T )

cpdf_2015_2.max()

cpdf_2015_2_nm = cpdf_2015_2/cpdf_2015_2.max()

import sklearn 

sklearn.metrics.pairwise.euclidean_distances?


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












cpdf_2015_2.values, 




# for i in l2:
#     if i in l1:
#         print('trut')
#     else:
#         print( i )







# for tmp_prd in prd_iniq:
#     for tmp_cnt in cnt_uniq:
#         for tmp_ind in ind_uniq:





