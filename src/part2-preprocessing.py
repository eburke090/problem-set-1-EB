'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages
import pandas as pd
import os


# Your code here

#load csv's into dataframes
pred_universe = pd.read_csv('data/pred_universe_raw.csv', parse_dates=['arrest_date_univ'])
arrest_events = pd.read_csv('data/arrest_events_raw.csv', parse_dates=['arrest_date_event'])

#full outer join/merge
df_arrests = pd.merge(pred_universe, arrest_events, how='outer', on='person_id')
#check if column exists
df_arrests = df_arrests.dropna(subset=['arrest_date_univ'])

#creating y
#sort for useable data
arrest_events = arrest_events.sort_values(by='arrest_date_event')
#check if felony arrest occured within 1 year after current arrest
def was_rearrested(person_id, arrest_date):
    if pd.isnull(arrest_date):
        return 0
    window_start = arrest_date + pd.Timedelta(days=1)
    window_end = arrest_date + pd.Timedelta(days=365)
    future_felonies = arrest_events[
        (arrest_events['person_id'] == person_id) &
        (arrest_events['arrest_date_event'] >= window_start)&
        (arrest_events['arrest_date_event'] <= window_end) &
        (arrest_events['offense_category'] =='F')
    ]
    return int(not future_felonies.empty)

df_arrests['y'] = df_arrests.apply(
    lambda row: was_rearrested(row['person_id'], row['arrest_date_univ']), axis=1
)
#print statement
share_rearrested = df_arrests['y'].mean()
print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {share_rearrested:.3f}")

#create current_charge_felony feature
df_arrests['current_charge_felony'] = df_arrests['offense_category'].apply(lambda x: 1 if x == 'F' else 0)
#print statement
share_felony_current = df_arrests['current_charge_felony'].mean()
print(f"What share of current charges are felonies? {share_felony_current:.3f}")


#create num_fel_arrests_last_year feature
def count_prior_felonies(person_id, arrest_date):
    if pd.isnull(arrest_date):
        return 0
    window_start = arrest_date -pd.Timedelta(days=365)
    window_end = arrest_date - pd.Timedelta(days=1)
    past_felonies = arrest_events[
        (arrest_events['person_id'] == person_id)&
        (arrest_events['arrest_date_event'] >= window_start) &
        (arrest_events['arrest_date_event'] <= window_end) &
        (arrest_events['offense_category'] == 'F')
    ]
    return len(past_felonies)

df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(
    lambda row: count_prior_felonies(row['person_id'], row['arrest_date_univ']), axis=1
)
#print statement + average felony
avg_prior_felonies = df_arrests['num_fel_arrests_last_year'].mean()
print(f"What is the average number of felony arrests in the last year? {avg_prior_felonies:.3f}")

#print mean of pred_universe
print("Mean of num_fel_arrests_last_year for pred_universe:")
print(df_arrests['num_fel_arrests_last_year'].mean())

#print preview
print("Preview of df_arrests:")
print(df_arrests.head())

#save for part 2
os.makedirs('data', exist_ok=True)
