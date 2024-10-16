
# %%
'''
Created by Richard Felius

Main script for the GRIP HAP project.
- Run this script to process raw data
- Get activity scores for every 10 seconds / minute
- creates an excel with outcomes per person per measurement (day)


# TODO


'''
import logging
import pandas as pd
from src.utils import *
from src.characteristics import characteristics, characteristics_pain
from src.prepare_data import prepare_data, split_data, clean_data
from src.predict_data import predict_data
import os

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'PAIN_SCORES': False,
    'RESULTS_2HOURS': False,
    'VERBOSE': True,
    'VISUALISE': True,
    'RAW_DATA_DIR': 'data/raw_data',
    'CONFIG_DIR': 'config'
}

# %%


def main():
    # Create folders and logging
    initialise_project()

    # Load config file and settings
    config = load_config(settings, config_name="config_file.yaml")
    subjects = os.listdir(settings['RAW_DATA_DIR'])
    final_results = pd.DataFrame()
    if settings['RESULTS_2HOURS']:
        final_results_2hours = pd.DataFrame()

    # Loop over subjecs
    for subject in subjects:
        if settings['VERBOSE']:
            print(f'Analysing subject {subject}')
        try:
            if subject.endswith('.DS_Store'):
                continue

            if settings['PAIN_SCORES']:
                # Load painscores
                painscores = pd.read_csv(
                    f'data/raw_data/{subject}/pain_score/{subject}_pain_score.csv', index_col=0)

            days = os.listdir(f"{settings['RAW_DATA_DIR']}/{subject}")
            # days.remove('pain_score')

            # Loop over days
            for day in days:
                if day.endswith('.csv'):
                    continue
                try:
                    if settings['VERBOSE']:
                        print(f'Analysing day {day}')
                    results = {}
                    file_path = os.listdir(
                        f"{settings['RAW_DATA_DIR']}/{subject}/{day}")[0]
                    file = f"{subject}/{day}/{file_path}"
                    file_name = f"{subject}_{day}"

                    # Load data, downsample and remove not worn
                    data_df, endtime, begintime = prepare_data(
                        file, config, settings)

                    data_df = split_data(
                        data_df, begintime, endtime, config)

                    # Drop first and last 30 seconds and drop not worn
                    data_df, not_worn_samples = clean_data(
                        data_df, config, all=True)

                    # Predict data based on a previously trained ML algorithm
                    data = predict_data(data_df, config, settings,
                                        file_name, file, config['chunk_size'])

                    results['subject'] = subject
                    results['day'] = day
                    results['Samples'] = len(data_df) + not_worn_samples
                    results['not_worn_samples'] = not_worn_samples
                    results['Name'] = file
                    results['begintime'] = begintime
                    results['endtime'] = endtime
                    results = characteristics(
                        results, data['activities [1min]'].values)

                    # Pain score data
                    if settings['PAIN_SCORES']:
                        try:
                            results = characteristics_pain(
                                painscores, results, day)
                        except Exception as e:
                            logging.error(f'pain: {subject} {day} {e}')

                    # data to dataframe
                    results_df = pd.DataFrame.from_dict(
                        results, orient='index').transpose()
                    for column in results_df.columns[7:]:
                        results_df[column] = pd.to_numeric(
                            results_df.loc[:, column]).round(3)
                    final_results = pd.concat((final_results, results_df))

                    # Predict data based on a previously trained ML algorithm
                    if settings['RESULTS_2HOURS']:
                        for key in data['time_2hours'].unique():
                            if key == '':
                                continue
                            tmp_data = data.loc[data['time_2hours'] == key]

                            results = {}
                            results['subject'] = subject
                            results['day'] = day
                            results['Samples'] = len(tmp_data)
                            results['not_worn_samples'] = not_worn_samples
                            results['Name'] = file
                            results['begintime'] = begintime
                            results['endtime'] = endtime
                            results['key'] = key
                            if settings['PAIN_SCORES']:
                                try:
                                    results = characteristics_pain(
                                        painscores, results, day, time=f"{key.split('_')[0]}:00")
                                except Exception as e:
                                    logging.error(f'pain: {subject} {day} {e}')

                            results = characteristics(
                                results, tmp_data['activities [1min]'].values)
                            results_df = pd.DataFrame.from_dict(
                                results, orient='index').transpose()
                            for column in results_df.columns[10:]:
                                results_df[column] = pd.to_numeric(
                                    results_df.loc[:, column]).round(3)
                            final_results_2hours = pd.concat(
                                (final_results_2hours, results_df))

                except Exception as e:
                    logging.error(f'day: {subject} {day} {e}')
        except Exception as e:
            logging.error(f'subject: {subject} {day} {e}')

    # Replace empty values with 0
    final_results = final_results.fillna(0)
    final_results.to_excel('Results/results_per_day.xlsx', index=False)
    final_results.loc[:, 'average_activity_level':].corr().to_excel(
        'Results/correlations_per_day.xlsx')
    if settings['RESULTS_2HOURS']:
        # Replace empty values with 0
        final_results_2hours = final_results_2hours.fillna(0)
        final_results_2hours.to_excel(
            'Results/results_per_2hours.xlsx', index=False)


if __name__ == '__main__':
    main()

# %%
