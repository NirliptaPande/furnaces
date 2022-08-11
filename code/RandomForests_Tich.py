from datetime import datetime

startTime = datetime.now ()
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import os
import re
import pickle
import sys
import subprocess

test_features = []


def read_datacube(path_m):
    print ('Reading and filtering datasets...')
    #file1 = xr.open_dataset ('dataset_clean.nc', decode_times=True, decode_timedelta=False)

    # files = ['dataset_vars.nc','dataset_subset_2003-2011.nc','dataset_validation_2012-2016.nc']
    files = ['dataset_validation_2012-2016.nc']
    for raw_df in files:
        print (raw_df)
        file1 = xr.open_dataset (raw_df, decode_times=True, decode_timedelta=False)

        print ('Converting datasets to dataframe...')

        features = file1.to_dataframe ()  # convert dataset to pandas dataframe

        features.reset_index (inplace=True)  # reset index, move lon and lat to columns form indices

        pd.set_option ('display.expand_frame_repr', False)  # show all column in pandas df

        print (features.head (50))  # check columns are displayed properly

        # Filters:
        # features =features[features.topo==1] #by topography to remove all ocean values
        # features =features[features.lat>-56] #latitudes below 56S
        features = features.replace (-9997, 0)
        features = features.replace (-9999, 0)
        # features = features[features.pft_fracCover > 0.3]  # remove non-burnable areas
        print ('check raw data structure:')

        features.columns = features.columns.str.replace (' days', '')
        print (features.head)

        # Save dataframes:

        if raw_df == 'dataset_validation_2012-2016.nc':

            pickle.dump (features, open ('dataset_validation_2012-2016.pkl', 'wb'), protocol=4)

            features = features[['lon', 'lat', 'time']]

            pickle.dump (features, open ('dataset_validation_2012-2016_markers.pkl', 'wb'), protocol=4)



        elif raw_df == 'dataset_subset_2003-2011.nc':

            pickle.dump (features, open ('dataset_subset_2003-2011.pkl', 'wb'), protocol=4)

        elif raw_df == 'dataset_vars.nc':

            pickle.dump (features, open ('dataset_clean.pkl', 'wb'), protocol=4)


def read_df(path_m, file):
    # open a file, where you stored the pickled data

    with open (file, 'rb') as fr:
        features = pickle.load (fr)

    print ('file loaded')

    return features


def quality_check(path_m, file):
    # read in saved dataframe

    features = read_df (path_m, file)

    print ('data quality check:')

    print ('cleaning data...')

    # Saving feature names for later use
    headers = list (features.columns)
    headers.remove ('lon')
    headers.remove ('lat')
    headers.remove ('time')

    if 'NLDI' in features.columns:
        features['NLDI'] = features['NLDI'].fillna (1.01)

    for i in headers:
        features[i] = features[i].fillna (0)

    features['wet'] = features['wet'].astype (int)
    print(features)
    return features, headers


def subsample(features):
    ###################################################
    # sub-sample features
    #########################################################
    # save the zero and non zero values to seperate dataframes
    ##########################################################
    f_zeros = features.loc[features['ign'] == 0]
    f_nonzeros = features.loc[features['ign'] != 0]

    # subsample a proportional number of non fire cells form the df for the model
    if f_nonzeros.shape[0] < f_zeros.shape[0]:

        xnum = f_nonzeros.shape[0]
    else:
        xnum = f_zeros.shape[0]

    non_zeros_filter = f_nonzeros.sample (n=xnum)  # add all ignitions

    zeros_filter = f_zeros.sample (n=xnum)  # add more zero ignitions than actual ignitions

    frames = [non_zeros_filter, zeros_filter]

    sampled_features = pd.concat (frames)

    sampled_features = sampled_features.sample (frac=1, random_state=42)

    return sampled_features


def regressor(train_features, train_labels, test_features, test_labels, headers, features, file, modelType):


    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor (n_estimators=500, criterion='mse', random_state=42, max_features=7, n_jobs=60,
                                   oob_score=True,
                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0.0, bootstrap=True, verbose=0,
                                   warm_start=False, ccp_alpha=0.0, max_samples=None)

    print ('fit random forest regressor model')

    # Train the model on training data

    model.fit(train_features, train_labels);

    print ('save model objects')

    rf_model_objects = (model, train_features, test_features, train_labels, test_labels, headers)
    os.chdir(path_m)


    pickle.dump (rf_model_objects, open ('rf' + inter + '_rr_val.pkl', 'wb'), protocol=4)


def remove_features_val(sampled_features: object, biomes, features, file, rm_list, modelType):
    print ('removing features')

    # Remove unused features

    rm_List = list(rm_list.split(","))

    rm_list = rm_List
    print(rm_list)

    for i in rm_list:
        del sampled_features[i]


    print('cleaned columns')
    print (sampled_features.columns)

    sampled_features['ign'] = sampled_features['ign'].astype ('Int64')

    y_resampled_reg = sampled_features['ign']

    del sampled_features['ign']

    X_resampled_reg = sampled_features

    # Saving feature names for later use

    headers = list (features.columns)

    if 'lon' in headers:
        headers.remove ('lon')

    if 'lat' in headers:
        headers.remove ('lat')

    if 'time' in headers:
        headers.remove ('time')

    # Using Skicit-learn to split data into training and testing sets

    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    print ('starting data split')

    train_features, test_features, train_labels, test_labels = train_test_split (X_resampled_reg,
                                                                                 y_resampled_reg,
                                                                                 test_size=0.01,
                                                                                 random_state=42,
                                                                                 stratify=biomes)

    ###summary of data split
    print ('Regressor Training Features Shape:', train_features.shape)
    print ('Regressor Training Labels Shape:', train_labels.shape)
    print ('Regressor Testing Features Shape:', test_features.shape)
    print ('Regressor Testing Labels Shape:', test_labels.shape)
    # save training features:

    print ('final df')

    print (train_features.head)

    # Train the Randomforest

    regressor (train_features, train_labels, test_features, test_labels, headers, features, file, modelType)


def rm_features(var_r):
    """

    :rtype: object
    """
    path = '//tmukunga/Data/Code/01_Data_preprocessing/'

    os.chdir(path)

    a_file = open ('rm_features.txt', 'r')

    # initialise list of variables to add stripped strings from txt file
    list_of_vars = []

    # read vars from file and add to list:
    for i in a_file:
        stripped_line = i.strip ()

        list_of_vars.append (re.sub ('\s+', ',', stripped_line))
    print(var_r)
    print(list_of_vars)

    if var_r != 'all':

        if var_r in list_of_vars:
            list_of_vars.remove(var_r)

    elif var_r == 'base':
        pass

    else:

        all = ['pftCrop','Livestock', 'GDP', 'road_density', 'Distance_to_populated_areas', 'WDPA_fracCover', 'pop_density']

        for i in all:
            list_of_vars.remove(i)

    # convert list of var names to string:
    listToStr = ' '.join(map(str,list_of_vars))

    # replace empty spaces with commas in new string:

    listToStr = re.sub (" ", ",", listToStr)

    print ('list to strings')

    print (listToStr)

    os.chdir ('//tmukunga/Data/Data/01_Combined_Datasets/ready/processed/')

    return listToStr


def run_VIF(path, sampled_features):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # the independent variables set

    X = sampled_features

    # VIF dataframe
    vif_data = pd.DataFrame ()

    vif_data["feature"] = X.columns

    # calculating VIF for each feature

    vif_data["VIF"] = [variance_inflation_factor (X.values, i)
                       for i in range (len (X.columns))]

    print (vif_data)


def run_RFglobal_validation(path, file, modelType, filter_list):
    features, headers = quality_check (path, file)

    print (features.columns)

    print ('######### IMPORTED TRAINING FEATURES ###########')

    # remove limiter on features.head to show all columns.
    pd.set_option ('display.expand_frame_repr', False)

    print ('Processing global model...')

    features = features.dropna ()

    print ('filter complete')
    print (features.head (50))

    print ((features['ign'] == 0).astype (int).sum (axis=0))
    print (features['ign'].shape)
    print ((features['ign'] != 0).astype (int).sum (axis=0))

    if file != 'dataset_subset_2003-2011_residuals.pkl':

        sampled_features = subsample (features)
    else:
        sampled_features = features

    biomes = sampled_features['Biome']

    rm_list = filter_list

    remove_features_val (sampled_features, biomes, features, file, rm_list, modelType)


if __name__ == "__main__":
    path = '//tmukunga/Data/Data/01_Combined_Datasets/ready/processed/'

    os.chdir (path)  # mainpath

    #  Run the following function once to read cleaned netcdf file, convert object to pandas df and save
    # path = mainpath+directory

    #read_datacube(path)

    modelType = 'rf'
    models = ['21','12', '13', '15', '16', '17', '18', '20']
    #vars to remove from text file
    #rm_vars = ['base','Livestock', 'GDP', 'road_density', 'Distance_to_populated_areas', 'WDPA_fracCover', 'pop_density', 'all']
    rm_vars = ['GDP', 'road_density', 'Distance_to_populated_areas', 'WDPA_fracCover', 'pop_density', 'all']

    for inter, rmv in zip (models, rm_vars):

        path_m = path + 'rf' + inter + '/'

        filter_list = rm_features(rmv)  # human variable to remove from model iteration

        Path (path_m).mkdir (parents=True, exist_ok=True)

        files = ['dataset_subset_2003-2011.pkl']

        np.seterr (divide='ignore')  # ignore divisions by zero
        for i in files:

            if i == 'dataset_subset_2003-2011.pkl':

                print ('x = 1')

                run_RFglobal_validation (path_m, i, modelType, filter_list)

            else:
                print (i + ' is not in file list')

    # run postprocessing script
    cmd = 'python postprocessing.py'
    p = subprocess.Popen (cmd, shell=True)
    out, err = p.communicate ()
    print (err)
    print (out)

    print (datetime.now () - startTime)
