
import pandas as pd
import numpy as np

name = 'country_indication_level_plc'

plc = pd.read_csv( name + '.csv' , header = [0,1] )



np.unique( [i[0] for i in plc.columns] )

a = [i[0] for i in plc.columns]

countries = [a[i] for i in sorted(np.unique(a, return_index=True)[1])][1:]

# plc.loc[:,('United Kingdom',)]

plct = plc.T
plct.columns  = plc.iloc[:,0].values

import copy

country_wise_frame = {}
country_ = 'United Kingdom'

for country_ in countries:
    print(country_)
    tmp_plctt = copy.deepcopy( plct.loc[( country_ ,)] ).T
    
    market_size_monthly = np.array( [i if i!=0 else 1 for i in tmp_plctt.sum(axis = 1).values[:-1]] )
    
    tmp_plctt_shr = pd.DataFrame(  tmp_plctt.values[:-1]/  market_size_monthly.reshape(-1,1) , index = tmp_plctt.index[:-1], columns = tmp_plctt.columns  )
    tmp_plctt_shr0 = pd.DataFrame(  tmp_plctt.values[:-1]  , index = tmp_plctt.index[:-1], columns = tmp_plctt.columns  )
    
    start_aligned_ts = {}
    
    tmp_plctt_shr_t = tmp_plctt_shr.T
    tmp_plctt_shr_t0 = tmp_plctt_shr0.T
    
    starting_point = 0.02
    
    for row_inx, row in enumerate( tmp_plctt_shr_t.values ):
        # row = tmp_plctt_shr_t.values[-2]
        # print( tmp_plctt_shr_t.index[row_inx] )
        t_row = []
        t_row_inx = []
        # print(row)
        for inx,rowv in enumerate(row):
            # print(inx)
            if rowv >=  starting_point:
                # t_row = row[inx:]
                t_row = tmp_plctt_shr_t0.values[row_inx][inx:]
                t_row_inx = tmp_plctt_shr_t.columns[inx:]
                
                break
        
        start_aligned_ts[( country_ , tmp_plctt_shr_t.index[row_inx] , 'values' )] = t_row
        start_aligned_ts[( country_ , tmp_plctt_shr_t.index[row_inx] , 'dates' )] = t_row_inx

            
# pd.DataFrame.from_dict( start_aligned_ts )


    required_frame = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in start_aligned_ts.items() ]))
    country_wise_frame[country_] = required_frame

rframes = [ v for k,v in country_wise_frame.items() ]
joined_frame = rframes[0]

for i in rframes[1:]:
    joined_frame = joined_frame.join( i, how= 'outer' )


# joined_frame.to_csv( 'aligned_ts.csv' )
# required_frame.to_csv( 'UK_aligned_ts.csv' )


joined_frame_no_dates = copy.deepcopy( joined_frame.iloc[:, 0::2] )

joined_frame_no_dates_nan = copy.deepcopy( joined_frame_no_dates.dropna(axis=1, thresh= 24) )

joined_frame_no_dates_nan.fillna( 0 , inplace =True)

# joined_frame_no_dates_nan.fillna( -0.01 , inplace =True)

# # # joined_frame_no_dates_nan.to_csv( 'aligned_ts_no_dates_nan_country_lvl_tmp.csv' )

joined_frame_no_dates_nan.columns

joined_frame_no_dates_nan.loc[:, [('United Kingdom')]]

countries0 = copy.deepcopy(countries)
cntr_plc ={}

for cntry in countries0:
    # cntry = countries[0]
    tmp_frame = copy.deepcopy( joined_frame_no_dates_nan.loc[:, [(cntry)]] )
    cntr_plc[cntry] = tmp_frame.sum( axis =1).values

cntr_plc0  = pd.DataFrame.from_dict( cntr_plc )

cntr_plc0.values

series = np.array(cntr_plc0.values)

#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series)), average_series)
plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import matplotlib.pyplot as plt

aligned_ts_max_no_dates = pd.read_csv( 'country_lvl_aligned_ts_max.csv' , header = [0,1], index_col=0)

aligned_ts_max_no_dates1 = aligned_ts_max_no_dates.replace(to_replace = -0.01, value = 0)
# aligned_ts_max_no_dates1.drop( 'Unnamed: 0', axis=1, inplace= True )
# aligned_ts_max_no_dates1.drop( 0, axis = 0, inplace = True)
# aligned_ts_max_no_dates1.values

series = aligned_ts_max_no_dates1.iloc[:, 1:].T.values

series = [i/i.max() for i in series]

#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

# calculating average series with DBA
# average_series = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series)), average_series)
plt.show()






cntr_plc0 = {}

for cntry in countries0:
    # cntry = countries[0]
    tmp_frame = copy.deepcopy( plc.loc[:, [(cntry)]] )
    cntr_plc0[cntry] = tmp_frame.sum( axis =1).values

cntr_plc01 = pd.DataFrame.from_dict( cntr_plc0 )

series = cntr_plc01

series = series.ewm( 12 ).mean()
series = series.T.values

# series = cntr_plc01.T.values

series = [i/i.max() for i in series]


#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
# # average_series_unaligned = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_unaligned)), average_series_unaligned)
plt.show()








# agregate time-series max-aligned
# agregate time-series max-aligned

# aligned_ts_max_no_dates1.columns = [i[0] for i in aligned_ts_max_no_dates1.columns.values]

# series = aligned_ts_max_no_dates1.loc[:, [ 'Australia', 'Japan', 'Canada', 'Brazil' ]]
# series = aligned_ts_max_no_dates1.loc[:, [  'Brazil', 'Spain' ]]
series = aligned_ts_max_no_dates1.loc[:, [  'United Kingdom' ]]

# series = series.ewm( 12 ).mean()
series = series.T.values


series = [i/i.max() for i in series]

#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series_c1 = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_c1)), average_series_c1)
# plt.
plt.show()





# agregate time-series unaligned
# agregate time-series unaligned

series = cntr_plc01.loc[:, [  'Japan', 'Canada' ]]

series = series.ewm( 12 ).mean()
series = series.T.values

series = [i/i.max() for i in series]


#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series_unaligned_c1 = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_unaligned_c1)), average_series_unaligned_c1)
plt.show()




# agregate time-series max-aligned
# agregate time-series max-aligned


# aligned_ts_max_no_dates1.columns = [i[0] for i in aligned_ts_max_no_dates1.columns]

series = aligned_ts_max_no_dates1.loc[:, [ 'Australia', 'Finland', 'Canada', 'Spain', 'Switzerland' ]]

series = series.ewm( 12 ).mean()
series = series.T.values

series = [i/i.max() for i in series]

#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series_c1 = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_c1)), average_series_c1)
plt.show()




# agregate time-series unaligned
# agregate time-series unaligned

series = cntr_plc01.loc[:, [ 'Australia', 'Finland', 'Canada', 'Spain', 'Switzerland' ]]

series = series.ewm( 12 ).mean()
series = series.T.values

series

# series = [np.polyfit( list(range(len(i))) ,i , 8 ) for i in series ]

series = [i/i.max() for i in series]


#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series_unaligned_c1 = performDBA(series)

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_unaligned_c1)), average_series_unaligned_c1)
plt.show()


# agregate time-series unaligned
# agregate time-series unaligned

series = plc.loc[:, [ 'Australia', 'Finland', 'Canada', 'Spain', 'Switzerland' ] ]


series = series.ewm( 12 ).mean()
series = series.T.values

series

# series = [np.polyfit( list(range(len(i))) ,i , 8 ) for i in series ]

series = [i/i.max() for i in series]


#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
# # # average_series_unaligned_c1 = performDBA(series)
# # # 
# # # average_series_unaligned_c1_copy = copy.deepcopy( average_series_unaligned_c1  )

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_unaligned_c1_copy)), average_series_unaligned_c1_copy)
plt.show()











# agregate time-series unaligned
# agregate time-series unaligned

series = plc.loc[:, [  'Canada', 'Spain' ] ]


series = series.ewm( 12 ).mean()
series = series.T.values

series

# series = [np.polyfit( list(range(len(i))) ,i , 8 ) for i in series ]

series = [i/i.max() for i in series]


#plotting the synthetic data
for s in series:
    plt.plot(range(0,len(s)), s)
plt.draw()

#calculating average series with DBA
average_series_unaligned_c1 = performDBA(series)
# # # 
# # # average_series_unaligned_c1_copy = copy.deepcopy( average_series_unaligned_c1  )

#plotting the average series
plt.figure()
plt.plot(range(0,len(average_series_unaligned_c1)), average_series_unaligned_c1)
plt.show()


