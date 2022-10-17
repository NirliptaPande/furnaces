from datetime import datetime

startTime = datetime.now()

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
import os
import re
from PyALE import ale
from netCDF4 import Dataset


def plot_histograms(path, y_resampled):

    # plot histogram of data distribution
    bins = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, np.inf)  # define edges
    n, bins, patches = plt.hist(x=y_resampled, bins=bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of fires')
    plt.ylabel('Frequency')
    plt.title('Histogram of Global fire atlas resampled ignition distribution')
    plt.savefig('resampled_Ignitions.png', bbox_inches='tight', dpi=1080)
    plt.show()
    plt.close()




def evaluate_global_predictions(rf, features, filter_list, path):
    '''

    :param rf: random forest regressor
    :param features: raw data used to predict global monthly ignitions
    :return:
    '''
    fillnas = list(features.columns)
    for i in fillnas:
        features[i] = features[i].fillna(0)

    # remove limiter on features.head to show all columns.
    pd.set_option('display.expand_frame_repr', False)

    markers = features[['lat', 'lon', 'time']]

    global_ignitions = features['ign']
    # Remove unused features
    rm_list = filter_list

    rm_List = list(rm_list.split(","))

    rm_list = rm_List

    print('removed features:')

    print(rm_list)
    for i in features.columns:
        print(i)
    print(features.index)
    for i in rm_list:
        del features[i]

    print('cleaned table:')
    print(features.columns)

    biome_features = features

    print(biome_features)

    x = (biome_features, global_ignitions)

    del features['ign']

    Global_predictions = rf.predict(features)

    with open(model_dir + 'glob_predictions.pkl', 'wb') as f:
        pickle.dump(Global_predictions, f)
        f.close()
    with open(model_dir + 'markers.pkl', 'wb') as f:
        pickle.dump(markers, f)
        f.close()

    os.chdir(path)

    pickle.dump(Global_predictions, open('glob_predictions.pkl', 'wb'), protocol=4)

    pickle.dump(markers, open('markers.pkl', 'wb'))

    return global_ignitions, Global_predictions


def rm_features(var_r):
    rm_path = '//tmukunga/Data/Code/01_Data_preprocessing/'

    os.chdir(rm_path)
    
    a_file = open('rm_features.txt', 'r')

    # initialise list of variables to add stripped strings from txt file
    list_of_vars = []
    # read vars from file and add to list:
    for i in a_file:
        stripped_line = i.strip()
        # print(stripped_line)
        list_of_vars.append(re.sub('\s+', ',', stripped_line))

    if var_r != 'all':

        if var_r in list_of_vars:
            list_of_vars.remove(var_r)

    elif var_r == 'base':
        pass

    else:

        all = ['pftCrop','Livestock', 'GDP', 'road_density', 'Distance_to_populated_areas', 'WDPA_fracCover', 'pop_density']


        for i in all:
            list_of_vars.remove (i)

    # convert list of var names to string:
    listToStr = ' '.join(map(str, list_of_vars))
    # replace empty spaces with commas in new string:
    listToStr = re.sub(" ", ",", listToStr)
    print(listToStr)
    os.chdir('//tmukunga/Data/Data/01_Combined_Datasets/ready/processed/')
    return listToStr


def global_xr(rf, data_val, headers, raw_markers, filter_list, main_path):


    global_ignitions, global_predictions = evaluate_global_predictions(rf, data_val,filter_list, main_path)

    os.chdir(model_dir)

    markers = pickle.load(open('markers.pkl', 'rb'))

    ##=================================================
    y_true = global_ignitions
    y_pred = global_predictions

    new_df = pd.DataFrame({'Pred': y_pred, 'Obs': y_true})
    print(new_df.head(50))

    #normalised mean square arror
    mean_val = new_df['Obs'].loc[new_df['Obs']!=0].mean()
    print('x_bar nmse')
    print(mean_val)
    num_ = 0
    den = 0

    num_ = sum((new_df['Pred'].to_numpy () - new_df['Obs'].to_numpy ())**2)
    den = sum(((new_df['Obs'].to_numpy () - mean_val)**2))

    NMSE = num_/den
    print(inter)
    print('nmse:')
    print(NMSE)
    print('Correlation:')
    correlation = new_df['Pred'].corr(new_df['Obs'])
    print(correlation)


    ## return None
    #=====================================================================================
    print(markers.head(50))
    df_diff = pd.concat([markers, raw_markers]).drop_duplicates(keep=False)
    print(df_diff.head(20))
    print(df_diff.shape)

    df_diff['Ignition_predictions'] = np.nan

    # print(global_predictions)

    # time = pd.to_datetime(time_, format='%Y%m')
    # glob_predictions = pd.DataFrame(global_predictions, columns = ['predictions'])
    markers.reset_index(inplace=True)
    markers['Ignition_predictions'] = global_predictions
    markers = markers.append(df_diff)
    print(markers['time'])
    markers['time'] = pd.to_datetime(markers['time'], format="%Y%m")
    print(markers)

    # del markers['time']
    markers.set_index(['time', 'lat', 'lon'], inplace=True)
    del markers['index']
    print(markers)
    xr_data = markers.to_xarray()
    print(xr_data)

    xr_data.to_netcdf('predictions_ts.nc')

    return

def get_importance(model, headers):

    df_feature_importance = pd.DataFrame(model.feature_importances_, index=headers,
                                         columns=['feature importance']).sort_values('feature importance',
                                                                                     ascending=False)
    print(df_feature_importance)
    # feature importance in each tree

    df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in model.estimators_], columns=headers)
    df_feature_all.head()

    # melted data
    df_feature_long = pd.melt(df_feature_all, var_name='feature name', value_name='values')

    # Bar chart:
    df_feature_importance.plot(kind='bar')
    plt.savefig('rf_manual_resample_feature_importance_01.png', bbox_inches='tight', dpi=1080)
    plt.close()

    # Get numerical feature importances
    importances = list(model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round (importance, 2)) for feature, importance in zip (headers, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted (feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print ('Variable: {:20} Importance: {}'.format (*pair)) for pair in feature_importances]


    #list of x locations for plotting
    x_values = list(range(len(feature_importances)))
    # List of features sorted from most to least important
    sorted_importances = [importance[1] for importance in feature_importances]

    print(sorted_importances)
    new_list = [float (i) for i in sorted_importances]
    print (sorted_importances)
    sorted_features = [importance[0] for importance in feature_importances]

    # Function to convert
    def listToString(s):
        # initialize an empty string
        str1 = " "

        # return string
        return (str1.join (s))

    list = listToString (new_list)
    print (listToString (new_list))
    # Cumulative importances
    s_importances = new_list.astype(float)


    cumulative_importances = np.cumsum (s_importances)
    # Make a line graph
    plt.plot (x_values, cumulative_importances, 'g-')
    # Draw line at 95% of importance retained
    plt.hlines (y=0.95, xmin=0, xmax=len (new_list), color='r', linestyles='dashed')
    # Format x ticks and labels
    plt.xticks (x_values, sorted_features, rotation='vertical')
    # Axis labels and title
    plt.xlabel ('Ignitions');
    plt.ylabel ('Cumulative Importance');
    plt.title ('Cumulative Importances');
    plt.savefig ('Cumulative_importance.png', bbox_inches='tight', dpi=1080)

def plot_ALE(rf, train_features, path, rmv):
    os.chdir(path)
    print('restore tuple')
    # Restore tuple

    #headers = ['GDP','pop_density','Livestock','Distance_to_populated_areas','WDPA_fracCover']
    train_features = train_features.reset_index ()
    del train_features['index']
    print(train_features)
    res = isinstance (rmv, str) #check if variable is a string
    h = rmv
    print(h)
    if str(res) == 'False':

        fig, (ax1) = plt.subplots(1, figsize=(15, 7))
        #ale_res_1 = ale(X=train_features, model=rf, feature=[h], feature_type='continuous', grid_size=100,
        #                include_CI=True, C=0.95, plot=True, fig=fig, ax=ax1)
        ale_res_1 = ale (X=train_features, model=rf, feature=[h], feature_type='continuous',
                         grid_size=100,
                         fig=fig, ax=ax1)
        print(rmv)
        print(ale_res_1)
        ale_res_1.to_csv(r'aledata2d.txt',sep=' ', mode='a')

    else: # for baseline model
        cols = train_features.columns
        print(cols)
        print('computing ALEs')
        for c in cols:

            ale_res_1 = ale (X=train_features, model=rf, feature=[c], feature_type='continuous',
                         grid_size=100)
            print(c)
            print(ale_res_1)
            ale_res_1.to_csv(c+'_caledata2d.txt',sep=' ', mode='a')
        return



# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    #define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=60)
    return scores

def calc_quantiles(train_features):

    cols = ['pftCrop','road_density','WDPA_fracCover','Livestock','pop_density','Distance_to_populated_areas','GDP']

    sub_data = train_features[cols]

    q_005 = sub_data.quantile(0.05)
    q_025 = sub_data.quantile(0.25)
    q_05 = sub_data.quantile(0.5)
    q_075 = sub_data.quantile(0.75)
    q_095 = sub_data.quantile(0.95)

    print(q_005)
    print(q_025)
    print(q_05)
    print(q_075)
    print(q_095)

def crossval(model, train_features, train_labels):

    # define dataset
    X, y = train_features, train_labels
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        # evaluate the model
        scores = evaluate_model(model, X, y)
        # store the results
        results.append(scores)
        names.append(name)
        # summarize the performance along the way
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.savefig(model + 'CV_eval.png', bbox_inches='tight', dpi=2160)
    plt.show()

if __name__ == "__main__":

    main_path = '//tmukunga/Data/Data/01_Combined_Datasets/ready/processed/'
    os.chdir(main_path)
    raw_markers = pickle.load(open('dataset_validation_2012-2016_markers.pkl', 'rb'))

    modelType = 'rf'

    #models = ['rf22','rf21','rf11','rf12', 'rf13', 'rf15', 'rf16', 'rf17', 'rf18', 'rf20']
    models = ['rf22']

    #rm_vars = ['pastr','base','pftCrop','Livestock', 'GDP', 'road_density', 'Distance_to_populated_areas', 'WDPA_fracCover', 'pop_density', 'all']
    rm_vars = ['pastr']
    for inter, rmv in zip (models, rm_vars):

        model_dir = main_path + inter + '/'

        filter_list = rm_features(rmv)  # human variable to remove from model iteration

        model_main = model_dir + inter + '_rr_val.pkl'

        print('Entered:', model_main)

        r_model_val = inter + '_rr_val.pkl'
        print('Validation model:', r_model_val)

        # read datasets
        val_data = 'dataset_validation_2012-2016.pkl'

        try:
            os.chdir(main_path)
            data_val = pickle.load(open(val_data, 'rb'))

        except IOError:
            print('Validation data file does not exist!')

        try:
            os.chdir(model_dir)
            rf_val, train_features, test_features, train_labels, test_labels, headers = pickle.load(
                open(r_model_val, 'rb'))  # base model
        except IOError:
            print('main model object file does not exist!')

        pd.set_option('display.expand_frame_repr', False)

        headers = train_features.columns
        print('headers in baseline')
        print(headers)



        #get_importance(rf, headers)
        global_xr(rf_val, data_val, headers, raw_markers, filter_list, model_dir)
        #plot_ALE (rf_val, train_features, model_dir, rmv)
        #calc_quantiles(train_features)

        print(datetime.now() - startTime)
