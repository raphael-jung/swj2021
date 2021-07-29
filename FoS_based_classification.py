import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
import utils
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import time
import h2o
from h2o.automl import H2OAutoML


def prepare_pegonet_data(years_to_record, number_of_pegonets_to_search, columns):
    def get_graph(conn):
        c = conn.cursor()
        Gs = {}
        for year in years_to_record:
            # Get the graph for the given iteration.
            G = nx.Graph()

            # Get list of unique nodes which were used in the given year,
            c.execute(f'''
                SELECT DISTINCT PaperFieldsOfStudy.FieldOfStudyId, {year} - FirstYear
                FROM PaperFieldsOfStudy, FieldsOfStudy
                WHERE 
                    PaperFieldsOfStudy.FieldOfStudyId = FieldsOfStudy.FieldOfStudyId AND
                    PaperFieldsOfStudy.Year = {year} 
                ''')
            for n, age in c.fetchall():
                G.add_node(n, age=age)

            # Get list of fos combinations in the given year, with frequency as weight
            c.execute(f'''
                SELECT F1.FieldOfStudyId, F2.FieldOfStudyId, count(P.PaperId)
                FROM 
                    (SELECT DISTINCT PaperId FROM PaperFieldsOfStudy WHERE Year = {year}) AS P,
                    PaperFieldsOfStudy AS F1,
                    PaperFieldsOfStudy AS F2
                WHERE 
                    F1.PaperId = P.PaperId AND 
                    F2.PaperId = P.PaperId AND 
                    F1.FieldOfStudyId < F2.FieldOfStudyId 
                GROUP BY F1.FieldOfStudyId, F2.FieldOfStudyId
                ''')
            for f, t, freq in c.fetchall():
                G.add_edge(f, t, weight=freq, distance=1.0/freq)
            Gs[year] = G
        return Gs

    def record_pegonet_properties(Gs, main_type, fos_name, year, number_of_pegonets_to_search, egonet_fname, columns):
        def get_pegonets(Gs, year, number_of_pegonets_to_search):
            # Divide new and old fos, at year+1
            seeds_new, seeds_old = \
                [s for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0], \
                [s for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0]

            # get egonets of new and old fos, at year+1
            pegonets_new = sorted([(Gs[year+1].nodes[seed]['age'] == 0, seed, {node for node, edges in Gs[year+1][seed].items()}) for seed in seeds_new], key = lambda x: len(x[2]), reverse=True)
            pegonets_old = sorted([(Gs[year+1].nodes[seed]['age'] == 0, seed, {node for node, edges in Gs[year+1][seed].items()}) for seed in seeds_old], key = lambda x: len(x[2]), reverse=True)

            # Filter the egonets to the desired number - there could be a large size discrepancies, so remove larger parts first before the actual filteration
            newmax = max([len(i[2]) for i in pegonets_new])
            oldmax = max([len(i[2]) for i in pegonets_old])
            if newmax >= oldmax:
                new = [i for i in pegonets_new if len(i[2]) <= oldmax][:round(number_of_pegonets_to_search/2)]
                old = pegonets_old[:round(number_of_pegonets_to_search/2)]
            else:
                new = pegonets_new[:round(number_of_pegonets_to_search/2)]
                old = [i for i in pegonets_old if len(i[2]) <= newmax][:round(number_of_pegonets_to_search/2)]

            return new + old

        outcome = []
        # Exit if something went wrong
        egonets = get_pegonets(Gs, year, number_of_pegonets_to_search)
        if len(egonets) == 0:
            print("pegonet either only true/false, or nonexistent, ", sep=' ', end='', flush=True)
            return None

        # Get Pegonets at year
        for isnewfos, seed, egonet in egonets:
            subgraph = nx.Graph(Gs[year].subgraph(egonet))
            
            # skip if the whole subgraph is just empty
            if len(subgraph.edges())==0 or len(subgraph.nodes())==0: continue 
        
            if len(subgraph.edges())>0:
                r = [
                    main_type,
                    fos_name,
                    year,
                    seed,
                    Gs[year+1].nodes[seed]['age']-1, #the NEW fos in y+1 would have no 'age' in y because it didn't exist back then.
                    isnewfos]\
                    + utils.getstats(subgraph, Gs[year], egonet)
                outcome.append(r)
            else:
                outcome.append([main_type,fos_name,year,seed,Gs[year+1].nodes[seed]['age']-1,isnewfos] + ['-']*len(properties))

        df = pd.DataFrame(outcome)
        df.to_csv(egonet_fname, sep='\t', index=False, mode='a', header=None)
        return None

    for main_type, fos_name in utils.get_fos_list():
        # Make connection to the dataset
        conn = utils.connect(main_type, fos_name)

        # get the file name to store the iteration result
        egonet_fname = f'./egonets/{main_type}__{fos_name}.csv'
            
        # Check the db - if it's already recorded, skip this iteration.
        if utils.check_file_exists(egonet_fname):
            print(f"{main_type}, {fos_name}: Already done - skipping.")
            continue
        # Create file headers if this is a new file.
        else:
            with open(egonet_fname, 'w') as f:
                f.write('\t'.join(columns)+'\n')

        # Go over the given years to get the graph - 2000 ~ 2020 for now.
        Gs = utils.measure_time(get_graph, f"Building pre-zoomed graph for {main_type}, {fos_name}", (conn,))

        # Go over all but the last year - the last year is not used because year+1 is used to identify pegonets!
        for year in years_to_record[:-1]:
            # Get egonets and store their properties
            utils.measure_time(record_pegonet_properties, f"Recording egonet properties for {main_type}, {fos_name}, {year}", (Gs, main_type, fos_name, year, number_of_pegonets_to_search, egonet_fname, columns))

        # Close the connection
        utils.close(conn)

@ignore_warnings(category=ConvergenceWarning)
def classify_egonet_within_fos(outfname, group_by, properties, training_set_years_list, targetLabel, shuffleData=True):
    outcome = []
    group_by.remove('year') # year is used for training data
    
    for training_set_years in training_set_years_list:
        print(training_set_years)    
        for main_type, fos_name in utils.get_fos_list():
            # set the classifier. 
            c = LogisticRegression(solver='lbfgs', max_iter=100)

            # Get the dataframe
            df = utils.get_df(main_type, fos_name, do_shuffle=shuffleData)

            # Loop between the years 
            for year_test in range(df['year'].min() + training_set_years, df['year'].max()+1):
                
                trueCount = len(df.loc[(df[targetLabel] == True) & (df['year'] == year_test)])
                falseCount= len(df.loc[(df[targetLabel] == False) & (df['year'] == year_test)])

                Data_train = df.loc[(df['year'] < year_test) & (df['year'] >= year_test - training_set_years)]
                if len(Data_train.index)==0:
                    print("Training data empty, skipping", main_type, fos_name, year_test)
                    continue
                X_train = Data_train[properties]
                Y_train = Data_train[targetLabel]
                
                Data_test = df.loc[df['year'] == year_test]
                if len(Data_test.index)==0:
                    print("Test data empty, skipping", main_type, fos_name, year_test)
                    continue
                X_test = Data_test[properties]
                Y_test = Data_test[targetLabel]

                # fit data and store result
                o = utils.getAcc(c, X_train, Y_train, X_test, Y_test)
                outcome.append(['Not Standardized', training_set_years, main_type, fos_name, year_test] + o)

                # Standardize the data
                X_train_standardized, X_test_standardized = utils.standardize(X_train, X_test)                
                o = utils.getAcc(c, X_train_standardized, Y_train, X_test_standardized, Y_test)
                outcome.append(['Standardized', training_set_years, main_type, fos_name, year_test] + o)
        
    pd.DataFrame(outcome, columns=
        ['is_standardized', 'training_years', 'main_type', 'fos', 'year', 'ROCscore', 'Acc', 'Precision', 'Recall', 'F1score', 'TP', 'FP', 'TN', 'FN'])\
        .to_csv(outfname, sep='\t', index=False)


@ignore_warnings(category=ConvergenceWarning)
def classify_egonet_within_fos_pca(outfname, group_by, properties, training_set_years_list, targetLabel, shuffleData=True, num_features=5):
    outcome = []
    
    for training_set_years in training_set_years_list:
        print(training_set_years)    
        for main_type, fos_name in utils.get_fos_list():
            # set the classifier. 
            c = LogisticRegression(solver='lbfgs', max_iter=100)

            # Get the dataframe
            df = utils.get_df(main_type, fos_name, do_shuffle=shuffleData)

            # Loop between the years 
            # for year_test in range(df['year'].min() + training_set_years, df['year'].max()+1):
            for year_test in range(1999,2019):
                
                data = df.loc[(df['year'] <= year_test) & (df['year'] >= year_test - training_set_years)]

                data_p = data[properties]
                data_l = data[[targetLabel,]].to_numpy()

                # feature selection
                X_cols = SelectKBest(mutual_info_classif, k=num_features).fit(data_p, data_l).get_support(indices=True)
                data_p = data_p[data_p.columns[X_cols]]

                # standardize the data
                m = np.mean(data_p, axis=0)
                std = np.std(data_p, axis=0)

                data_p_standardized = (data_p - m) / std
                
                pca = PCA(n_components=2)
                pc_arr = np.concatenate([
                    data_l,pca.fit_transform(data_p_standardized)],axis=1)
                
                pc_df = pd.DataFrame(pc_arr, columns=['label', 'pca_1', 'pca_2'])

                pca_variance = list(pca.explained_variance_ratio_) + [1.0-np.sum(pca.explained_variance_ratio_)]
                outcome.append([num_features, training_set_years, main_type, fos_name, year_test] + pca_variance)

                # # make a color map of fixed colors
                # cmap = colors.ListedColormap(['r', 'g'])

                # plot = pc_df.plot.scatter(x='pca_1', y='pca_2', c='label', colormap=cmap)
                
                # plot.title.set_text(f'PCA for {fos_name} in {year_test}, using {training_set_years} years')

                # fig = plot.get_figure()
                # fig.savefig(f"./fig/{fos_name}_{year_test}_{training_set_years}.png")
                
                # plt.close(fig)

    pd.DataFrame(outcome, columns=
        ['num_features', 'training_years', 'main_type', 'fos', 'year', 'ratio_explained_by_pca_1', 'ratio_explained_by_pca_2', 'unexplained_ratio']).to_csv(outfname, index=False, header=False, mode='a')


@ignore_warnings(category=ConvergenceWarning)
def classify_egonet_within_fos_pca_for_paper(outfname, group_by, properties, training_set_years_list, targetLabel, shuffleData=True, num_features=15, year_test=2010):
    outcome = []
    
    for training_set_years in training_set_years_list:
        print(training_set_years)    
        for main_type, fos_name in utils.get_fos_list():
            # set the classifier. 
            c = LogisticRegression(solver='lbfgs', max_iter=100)

            # Get the dataframe
            df = utils.get_df(main_type, fos_name, do_shuffle=shuffleData)

            # Use one year
            data = df.loc[(df['year'] <= year_test) & (df['year'] >= year_test - training_set_years)]

            data_p = data[properties]
            data_l = data[[targetLabel,]].to_numpy()

            # feature selection
            X_cols = SelectKBest(mutual_info_classif, k=num_features).fit(data_p, data_l).get_support(indices=True)
            data_p = data_p[data_p.columns[X_cols]]

            # standardize the data
            m = np.mean(data_p, axis=0)
            std = np.std(data_p, axis=0)

            data_p_standardized = (data_p - m) / std
            
            pca = PCA(n_components=2)
            pc_arr = np.concatenate([
                data_l,pca.fit_transform(data_p_standardized)],axis=1)
            
            pc_df = pd.DataFrame(pc_arr, columns=['label', 'pca_1', 'pca_2'])

            pca_variance = list(pca.explained_variance_ratio_) + [1.0-np.sum(pca.explained_variance_ratio_)]
            outcome.append([num_features, training_set_years, main_type, fos_name, year_test] + pca_variance)

            # make a color map of fixed colors
            cmap = colors.ListedColormap(['r', 'g'])

            plot = pc_df.plot.scatter(x='pca_1', y='pca_2', c='label', colormap=cmap, colorbar=False, figsize=(10,10))
            plot.margins(0,0)
            plot.xaxis.set_label_text("")
            plot.yaxis.set_label_text("")
            fig = plot.get_figure()
            fig.savefig(f'./fig/fig_{fos_name}_{year_test}_{training_set_years}.png', bbox_inches = 'tight', pad_inches = 0.02)
            
            plt.close(fig)



@ignore_warnings(category=ConvergenceWarning)
def classify_egonet_within_fos_varying_features(outfname, group_by, properties, training_set_years_list, targetLabel, featuresToSelect, shuffleData=True):

    group_by.remove('year') # year is used for training data
    
    # Change the different feature selection functions.
    # zip(['f_classif', 'mutual_info_classif', 'chi2', 'f_regression', 'mutual_info_regression'],\
        #     [f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression]):
        #Chi2 not done because the standardized data have negative values.
    for func_name, score_func in \
        zip(['f_classif', 'mutual_info_classif', 'f_regression', 'mutual_info_regression'],\
            [f_classif, mutual_info_classif, f_regression, mutual_info_regression]):
        # Change the number of features
        for k in featuresToSelect:
            for training_set_years in training_set_years_list:
                outcome = []
                print(func_name, k, training_set_years)    
                for main_type, fos_name in utils.get_fos_list():
                    # set the classifier. 
                    c = LogisticRegression(solver='lbfgs', max_iter=100)

                    # Get the dataframe
                    df = utils.get_df(main_type, fos_name, do_shuffle=shuffleData)

                    # Loop between the years 
                    for year_test in range(df['year'].min() + training_set_years, df['year'].max()+1):
                        
                        trueCount = len(df.loc[(df[targetLabel] == True) & (df['year'] == year_test)])
                        falseCount= len(df.loc[(df[targetLabel] == False) & (df['year'] == year_test)])

                        Data_train = df.loc[(df['year'] < year_test) & (df['year'] >= year_test - training_set_years)]
                        if len(Data_train.index)==0:
                            print("Training data empty, skipping", main_type, fos_name, year_test)
                            continue
                        X_train = Data_train[properties]
                        Y_train = Data_train[targetLabel]
                        
                        Data_test = df.loc[df['year'] == year_test]
                        if len(Data_test.index)==0:
                            print("Test data empty, skipping", main_type, fos_name, year_test)
                            continue
                        X_test = Data_test[properties]
                        Y_test = Data_test[targetLabel]

                        # Standardize the data
                        X_train, X_test = utils.standardize(X_train, X_test)    

                        #Select k features
                        X_cols = SelectKBest(score_func, k=k).fit(X_train, Y_train).get_support(indices=True)
                        # Get data to show only k features
                        X_train_new = X_train[X_train.columns[X_cols]]
                        X_test_new = X_test[X_test.columns[X_cols]]

                        # fit data and store result
                        o = utils.getAcc(c, X_train_new, Y_train, X_test_new, Y_test)
                        outcome.append(['Selected', func_name, k, training_set_years, main_type, fos_name, year_test] + o + [','.join(X_train.columns[X_cols])])

                        # find result EXCLUDING the selected features, ONLY when the k is smaller than the number of selected features
                        if k < len(properties):
                            X_train_except = X_train.drop(X_train.columns[X_cols], axis=1)
                            X_test_except = X_test.drop(X_test.columns[X_cols], axis=1)                                    
                            o = utils.getAcc(c, X_train_except, Y_train, X_test_except, Y_test)
                            outcome.append(['Excluded', func_name, k, training_set_years, main_type, fos_name, year_test] + o + [','.join(X_train.columns[X_cols])])
                pd.DataFrame(outcome, columns=
                    ['used_features', 'feature_selection_func', 'num_of_features', 'training_years', 'main_type', 'fos', 'year', 'ROCscore', 'Acc', 'Precision', 'Recall', 'F1score', 'features'])\
                    .to_csv(outfname, sep='\t', index=False, header=None, mode='a')

if __name__ == "__main__":
    # set variables
    years_to_record = list(range(1990,2021))
    number_of_pegonets_to_search = 200
    training_set_years_list = [9]
    featuresToSelect = list(range(1,16))

    targetLabel = 'isNewFOS'
    classification_fname = './Classification_result_automl.csv'


    group_by = [
        'main_type', 
        'fos', 
        'year']
    info = [
        'seed', 
        'age', 
        'isNewFOS']
    properties = [
        '#nodes',
        '#edges',
        'avg_degree',
        'avg_degree_freqWeighted',
        'avg_n_age',
        'avg_e_weight',
        'avg_shortest_path',
        'avg_pagerank',
        'avg_degree_centrality',
        'avg_betweeness_centrality',
        'cohesion',
        'density',
        'transitivity',
        'average_clustering',
        'avg_triangles'
        ]
    columns = group_by + info + properties

    # # 1. Get measures for egonets
    prepare_pegonet_data(years_to_record, number_of_pegonets_to_search, columns)

    # 2. Train using n years, and use +1 year as test set
    # classify_egonet_within_fos(classification_fname, group_by, properties, training_set_years_list, targetLabel)
    # classify_egonet_within_fos_varying_features(classification_fname, group_by, properties, training_set_years_list, targetLabel, featuresToSelect, shuffleData=True)

    # group_by.remove('year') # year is used for training data
    # classify_egonet_within_fos_pca('./Classification_pca_result.csv', group_by, properties, training_set_years_list, targetLabel, num_features=5)
    # classify_egonet_within_fos_pca('./Classification_pca_result.csv', group_by, properties, training_set_years_list, targetLabel, num_features=10)
    # classify_egonet_within_fos_pca_for_paper('./Classification_pca_result.csv', group_by, properties, training_set_years_list, targetLabel, num_features=15, year_test=2010)