import pandas as pd
import numpy as np


# name = 'country_lvl_percent_growth'
# name = 'per_capita_mmarket_size'
# name = 'per_capita_gdp'
# name = 'number_products'
# name = 'no_of_biosimilars'
# name = 'market_share_by_country'
# name = 'market_share_top4'
# name = 'market_share_bio'
# name = 'hhi'
# name = 'market_share_top2'
# name = 'cnt_per_capita_percent_growth_nan_corrected'
name = 'number_bio_'

# country_lvl_percent_growth = pd.read_csv( name + '.csv' ).iloc[1:134:2]
# country_lvl_percent_growth = pd.read_csv( name + '.csv' , index_col = 'Unnamed: 0' ).iloc[:-1,:]
country_lvl_percent_growth = pd.read_csv( name + '.csv' )



row = country_lvl_percent_growth.values[0]
pdf = pd.DataFrame( np.transpose( np.array( [ [row[0]]*53 , country_lvl_percent_growth.columns[1:].tolist(), row[1:].tolist() ] ) ) )


for row in country_lvl_percent_growth.values[1:]:
    print(row) 
    tmpdf = pd.DataFrame( np.transpose( np.array( [ [row[0]]*53 , country_lvl_percent_growth.columns[1:].tolist(), row[1:].tolist() ] ) ) )
    pdf = pdf.append( tmpdf )
    
    
pdf.to_csv(  name+ 'formatted.csv' )





features = pd.read_csv('formatted_features__.csv')

features_ = features.replace( 0, np.nan )

cnt_grp = features.groupby( 'country' )

rfeat = []
for k,v in cnt_grp:
    print(k)
    # print(v)
    # v = cnt_grp.get_group( 'United Kingdom')
    # v.mad?
    print( v.ewm( com=9 , axis = 0 , ignore_na =True ).mean() )
    print( v.iloc[:,2:] )
    
    ewm0 = copy.deepcopy( v.ewm( com=9 , axis = 0 , ignore_na =True ).mean() )
    
    rfeat.append( [v.iloc[-3,0]] + [k] + ewm0.iloc[-3,:].values.tolist() )
    
    
rfeat1 = pd.DataFrame( rfeat , columns = features.columns  )
    
rfeat1.to_csv( 'country_lvl_agregate_features.csv' )