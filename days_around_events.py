import pandas as pd
import numpy as np

def days_around_events(exogen, before, after):
    '''
    The function enlargens the exogen data set containing events by custom dummy varaibles (columns) for days before and
    after events. The User defines arrays 'before' and 'after' which tell how many days before and after each event shall
    be included. The function then runs for the days before and then the days after. Note that the new columns are stored in
    DataFrames called 'df_before' and 'df_after' and only merged to the exogen DataFrame at the end in order to aviod issues
    such as days after columns being created for the days before. Thus wee keep the exogen DF as it it inputted till all
    loops are complete.
    The basic concept of both loops is the same. First we create the above mentioned DF for storage. We add the Data column
    which is latter on made the index. Its purpose is simply to be a column for merging. The loop starts by running threw every
    event. For each event we extract the number in the 'before' or 'after' array according to its column number. Thus the
    User inputs the number of days before and after in the arrays acording to the column order of the events.  With the number of
    days for each event an array from 1 till the numbers of days is created. It is then used to iterate. For each of the numbers
    in this array a column is created. Note that for the days before we add 'd' because we look at days in the future. Further
    we have to add 'd' zeros at the end as the last 'd' observations have not 'd' days in the future. For the days after
    we look in the past, thus we subtract 'd' and add 'd' zeros at the beginning.


    Parameters:
       
       exogen: the exogen variables in a pandas DataFrame format with each column being a variable and the time as its index
       
       before: Array [] of the length of the columns of exogen. Each number says how many days in the past are to be considered
               for the events. The numbers are to be arranged in the order of the columns of the events.
               
       after : Array [] of the length of the columns of exogen. Each number says how many days in the future are to be 
               considered for the events. The numbers are to be arranged in the order of the columns of the events.


    Return: the exogen array append with the designated days before and after as columns
    '''

    df_before = pd.DataFrame()
    df_before['date'] = exogen.index

    for event in exogen.columns:

        event_col_number = exogen.columns.get_loc(event)

        before_event = [i for i in range(1,before[event_col_number]+1)]
        for d in before_event:
            day_before = list()
            for i in range(0,len(exogen.index)-d): 
                    if exogen.iloc[i+d,exogen.columns.get_loc(event)] == 1:
                        day_before.append(1)
                    else:
                        day_before.append(0)

            day_before.append(np.zeros(d))
            day_before=np.concatenate(day_before,axis=None)
            df_before[str(str(d)+'_days_before_'+event)] = day_before

    df_before.index = df_before['date']

    del df_before['date']

    df_after = pd.DataFrame()
    df_after['date'] = exogen.index


    for event in exogen.columns:

        event_col_number = exogen.columns.get_loc(event)

        after_event = [i for i in range(1,after[event_col_number]+1)]
        for d in after_event:
            day_after = list()
            day_after.append(np.zeros(d))
            for i in range(d,len(exogen.index)): 
                    if exogen.iloc[i-d,exogen.columns.get_loc(event)] == 1:
                        day_after.append(1)
                    else:
                        day_after.append(0)

            day_after=np.concatenate(day_after,axis=None)
            df_after[str(str(d)+'_days_after_'+event)] = day_after

    df_after.index = df_after['date']

    del df_after['date']
    
    exogen = pd.merge(exogen, df_before, left_index=True, right_index=True)
    exogen = pd.merge(exogen, df_after, left_index=True, right_index=True)
    
    return exogen

