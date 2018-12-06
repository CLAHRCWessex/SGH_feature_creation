def create_timeseries_from_events(df, col_start, col_end, col_to_split=None, start=None, end=None,freq='H'):
        """
        Takes event based df and creates a df with dt index with counts of events in each time period.
        
        Input:
        df, df, with event data
        col_start, str, column name with datetime of the start of the event
        col_end, str, column ame with dateitme of the end of hte event
        start, str, datetime of when to start time series from. e.g. '3/1/2015'
        end, str, datetime of when to end timeseries.
        col_to_split, str, column name of optional sub-splitting of data.


        Output: df, with timeseries index with counts of events for each period.
        """
        import pandas as pd
        import numpy as np
        #from flosp import basic_tools

        #### set default vals if none given
        if start == None:
            start = df[col_start].min() # could use col_end here to ensure that all events are finished.
        if end == None:
            end = df[col_start].max()
        if col_to_split != None:
            #### get list to interate if a split column is given
            cols_clean = {}
            for i in df[col_to_split].unique():
                if type(i) == np.str:
                    j = basic_tools.tidy_string(i)
                else:
                    j = i
                cols_clean[i] = j

        #### create new df with timeseries index
        index = pd.DatetimeIndex(start=start,end=end,freq=freq) #make index
        occ = pd.DataFrame(index=index) # make df


        #### interat over each rowin occ df
        dftemp = df

        if col_to_split == None:
            for index, row in occ.iterrows():
                index_time = index
                #get df with patients currently within trust
                mask = (dftemp[col_start] < index_time) & (dftemp[col_end] > index_time)
                dftemp2 = dftemp[mask]
                # calc size of df
                occ.loc[index,'count_all'] = dftemp2.shape[0]

        if col_to_split != None:

            for index, row in occ.iterrows():
                index_time = index
                #get df with patients currently within trust
                mask = (dftemp[col_start] < index_time) & (dftemp[col_end] > index_time)
                dftemp2 = dftemp[mask]
                # calc size of df after queries
                for i in cols_clean:
                    #print(i)
                    filter_val = cols_clean[i]
                    occ.loc[index,filter_val] = dftemp2[dftemp2[col_to_split] == i].shape[0]

            # problem inmplementing when column names do not exist in occ.
            #occ['count_all'] = 0
            #for k in cols_clean:
            #    occ['count_all'] += occ[i]


        return(occ)
    
    
def tidy_string(i):
    """ take string and return new string without any confusing chaars"""

    #j = i.lower()
    j = i.strip()
    j = j.replace(' ','_')
    j = j.replace('.','_')
    j = j.replace('?','_')
    j = j.replace('&','_')
    j = j.replace('%','perc')
    j = j.replace('(','')
    j = j.replace(')','')
    j = j.replace(':','')
    j = j.replace(';','')
    j = j.replace('-','')
    j = j.replace('/','')
    return(j)


def make_callender_columns(df,column,prefix):
    """
    takes a datetime column and creates multiple new columns with: hour of day, day of week, month of year.
    Input
    df, df: dataframe
    column, str: name of datetime column to work from
    prefix, str: give prefix for new column names

    Output
    df, df: new df with additional columns with numerical indicators for callender vars.
    """
    #_core.message('Making callender columns from:  ' + column)

    dft = df # needed for cases which do not have missing datetime values

    if df[column].isnull().sum() != 0:
        warnings.warn('Some datetimes in your column are missing.')
        dft = df[df[column].notnull()] # create ref of df without rows that contain missing vals


    df[prefix + '_hour'] = dft[column].dt.hour.astype(int)
    df[prefix + '_dayofweek'] = dft[column].dt.dayofweek.astype(int)
    df[prefix + '_day'] = dft[column].dt.day.astype(int)
    df[prefix + '_month'] = dft[column].dt.month.astype(int)
    df[prefix + '_week'] = dft[column].dt.week.astype(int)
    df[prefix + '_dayofweek_name'] = dft[column].dt.weekday_name.astype(str)
    df[prefix + '_year'] = dft[column].dt.year.astype(int)
    df[prefix + '_date'] = dft[column].dt.date.astype(object)
    df[prefix + '_flag_wkend'] = (df[prefix + '_dayofweek'] >= 5).astype(int)
    return(df)