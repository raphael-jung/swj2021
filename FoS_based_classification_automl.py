import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
import utils
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import chain
from itertools import tee
from itertools import islice
import traceback
import logging
import datetime
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
        # def get_seeds_by_size(Gs, year, size_of_pegonets_to_search):
        #     # Divide new and old fos, at year+1
        #     seeds_new, sizes_new = \
        #         tee(([s,len(list(Gs[year+1].neighbors(s)))] for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0))
        #     seeds_old, sizes_old =  \
        #         tee(([s,len(list(Gs[year+1].neighbors(s)))] for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0))

        #     # Filter the egonets to the desired number - there could be a large size discrepancies, so remove larger parts first before the actual filteration
        #     newmax = max((c for _,c in sizes_new))
        #     oldmax = max((c for _,c in sizes_old))
        #     if newmax >= oldmax:
        #         new = (s for s,c in seeds_new if c <= oldmax)
        #         old = (s for s,c in seeds_old)
        #     else:
        #         new = (s for s,c in seeds_new)
        #         old = (s for s,c in seeds_old if c <= newmax)

        #     return chain(new, old)

        def get_seeds(Gs, year, number_of_pegonets_to_search):

            # Divide new and old fos, at year+1
            seeds_new = sorted(((s, len(Gs[year+1][s].items())) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0), key = lambda x: x[1], reverse=True)
            seeds_old = sorted(((s, len(Gs[year+1][s].items())) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0), key = lambda x: x[1], reverse=True)

            sizes_new = [len(Gs[year+1][s].items()) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0]
            sizes_old = [len(Gs[year+1][s].items()) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0]

            # Filter the egonets to the desired number - there could be a large size discrepancies, so remove larger parts first before the actual filteration
            # If there are no egonets to get the size from, then use last year's max.
            newmax = max(sizes_new) if len(sizes_new) > 0 else max([len(Gs[year][s].items()) for s in Gs[year].nodes() if Gs[year].nodes[s]['age'] == 0])
            oldmax = max(sizes_old) if len(sizes_old) > 0 else max([len(Gs[year][s].items()) for s in Gs[year].nodes() if Gs[year].nodes[s]['age'] != 0])

            if newmax >= oldmax:
                new = islice(((s,c) for s,c in seeds_new if c <= oldmax), number_of_pegonets_to_search)
                old = islice(seeds_old, number_of_pegonets_to_search)
            else:
                new = islice(seeds_new, number_of_pegonets_to_search)
                old = islice(((s,c) for s,c in seeds_old if c <= newmax), number_of_pegonets_to_search)

            return chain(new, old)

        outcome = []
        try:
            # Exit if something went wrong
            seeds = get_seeds(Gs, year, number_of_pegonets_to_search)
        except Exception as e:
            logging.error(traceback.format_exc())
            print("Error occurred while getting the seeds.", sep=' ', end='', flush=True)

        # Get Pegonets at year
        try:
            for seed, size_in_next_y in seeds:
                isnewfos = Gs[year+1].nodes[seed]['age'] == 0

                subgraph = nx.Graph(Gs[year].subgraph(Gs[year+1].neighbors(seed))) # the neighbor subgraph in y (the one which can actually be SEEN in the given year)
                subgraph_next_y = nx.Graph(Gs[year+1].subgraph(Gs[year+1].neighbors(seed))) # the neighbor subgraph in y+1 (the one which actually happened with the topic)
                egonet = subgraph.nodes()

                # skip if the whole subgraph is just empty, or below the given size limit
                if len(subgraph.edges())==0 or len(subgraph.nodes())==0: continue

                pre = [
                        main_type,
                        fos_name,
                        year,
                        seed,
                        Gs[year+1].nodes[seed]['age']-1, #the NEW fos in y+1 would have no 'age' in y because it didn't exist back then.
                        isnewfos,
                        size_in_next_y,
                        len(subgraph_next_y.edges()) # number of edges between the topic neighbors IN y+1 (in the future)
                    ]
                if len(subgraph.edges())>0:
                    outcome.append(pre + utils.getstats(subgraph, Gs[year], egonet))
                else:
                    outcome.append(pre + ['-']*len(properties))
        except Exception as e:
            logging.error(traceback.format_exc())
            print("Something went wrong. Maybe pegonet either only true/false, or nonexistent, ", sep=' ', end='', flush=True)
            return None
        df = pd.DataFrame(outcome)
        df.to_csv(egonet_fname, sep='\t', index=False, mode='a', header=None)
        return None

    for main_type, fos_name in utils.get_fos_list():
        # Make connection to the dataset
        conn = utils.connect(main_type, fos_name)

        # get the file name to store the iteration result
        egonet_fname = utils.get_egonet_fname(main_type, fos_name)

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


def classify_egonet_within_fos_automl(outfname, group_by, properties, training_set_years_list, targetLabel, label_size_list, years_to_run, num_models = 10, shuffleData=True):
    def getmid(model_id):
        s = model_id.split('_AutoML')[0]
        return s.split('__') if '__' in s else s.split('_')
    h2o.init()
    h2o.no_progress()

    # Get the unfinished fos_maintype_year iterations.
    total_count, unfinished_groups = utils.get_iterations_from_egonet(outfname, group_by, training_set_years_list, label_size_list, years_to_run)
    group_count = 0
    for fos_name, groups in unfinished_groups:
        # if utils.is_finished(outfname, group_by, (year_test, fos_name, training_set_years, label_size)):
        #     print(year_test, ' done, ', sep=' ', end='', flush=True)
        #     continue
        
        # Get the dataframe
        df = utils.get_df(None, fos_name, do_shuffle=shuffleData)
        lev, maintype = utils.get_level_mt(fos_name)
        print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n{datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name} : Starting..")

        for _, row in groups.iterrows():
            training_set_years = row['training_length']
            label_size = row['#_per_label']
            year_test = row['year']
            
            group_count += 1
            print(f"{group_count} / {total_count} : {datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name}, {training_set_years}, {label_size}, {year_test}")

            # Select the top label_size rows - the data is already shuffled when df is loaded
            # first, get data for the relevant years,
            # Second, group them by year / isNewFOS and select top n rows from true/false of each year
            Data_train = \
                df.loc[(df['year'] < year_test) & (df['year'] >= year_test - training_set_years)]\
                .groupby(['year', targetLabel]).head(label_size).reset_index(drop=True)
            training_frame = h2o.H2OFrame(Data_train)

            Data_test = \
                df.loc[df['year'] == year_test]\
                    .groupby(['year', targetLabel]).head(label_size).reset_index(drop=True)
            leaderboard_frame = h2o.H2OFrame(Data_test)

            # Run AutoML
            aml = H2OAutoML(max_models = num_models, seed = 1, nfolds=10, verbosity=None)
            aml.train(x = properties, y = targetLabel, training_frame=training_frame, leaderboard_frame=leaderboard_frame)

            outcome = [[fos_name, lev, maintype, year_test, training_set_years, label_size, i+1] + getmid(l[0]) + l[1:] for i,l in enumerate(h2o.as_list(aml.leaderboard, use_pandas=False)[1:])]

            pd.DataFrame(outcome).to_csv(outfname, sep='\t', index=False, mode='a', header=None)

            # Get variable importance of Stacked Ensemble metalearner, IF it is shown in leaderboard
            # Get model ids for all models in the AutoML Leaderboard
            # print(f"{datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name}, {training_set_years}, {label_size}, {year_test}: AutoML finished.")


            try:
                model_ids = aml.leaderboard['model_id'].as_data_frame().iloc[:,0]

                stackedensemble = []
                featureimportance = []
                for mid in model_ids:
                    name1, name2 = getmid(mid)
                    model = h2o.get_model(mid)

                    # get ensemble importance
                    if name1 == 'StackedEnsemble':
                        metalearner = h2o.get_model(model.metalearner().model_id)
                        stackedensemble.extend(
                            [[fos_name, lev, maintype, year_test, training_set_years, label_size, name2]
                             + getmid(name3)
                             + [v] for name3, v in list(metalearner.coef_norm().items())
                            ])
                    # get feature importance
                    else:
                        fim = h2o.get_model(mid).varimp(use_pandas=True)
                        fim.insert(0, 'mid', name2)
                        fim.insert(0, 'model', name1)
                        featureimportance.append(fim)

                df_stackedensemble = pd.DataFrame(stackedensemble)
                df_featureimportance = pd.concat(featureimportance)

                # add relative columns
                df_featureimportance.insert(0, '#_per_label', label_size)
                df_featureimportance.insert(0, 'training_length', training_set_years)
                df_featureimportance.insert(0, 'year', year_test)
                df_featureimportance.insert(0, 'main_type', maintype)
                df_featureimportance.insert(0, 'level', lev)
                df_featureimportance.insert(0, 'fos', fos_name)

                df_stackedensemble.to_csv("Automl_ensemble_coefficient.csv", sep='\t', index=False, mode='a', header=None)
                df_featureimportance.to_csv("Automl_feature_importance.csv", sep='\t', index=False, mode='a', header=None)
            except Exception as e:
                logging.error(traceback.format_exc())
                print("Something went wrong.")

            # print(f"{datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name}, {training_set_years}, {label_size}, {year_test}: feature recording finished.")

            # Clean the models
            h2o.remove_all()

    h2o.cluster().shutdown(prompt = False)



def get_intercepts(outfname, group_by, properties, training_set_years_list, targetLabel, label_size_list, years_to_run, num_models = 1, shuffleData=True):
    h2o.init()
    h2o.no_progress()

    # Get the unfinished fos_maintype_year iterations.
    total_count, unfinished_groups = utils.get_iterations_from_egonet(outfname, group_by, training_set_years_list, label_size_list, years_to_run)
    group_count = 0
    for fos_name, groups in unfinished_groups:
        # Get the dataframe
        df = utils.get_df(None, fos_name, do_shuffle=shuffleData)
        lev, maintype = utils.get_level_mt(fos_name)
        print(f"\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n{datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name} : Starting..")

        for _, row in groups.iterrows():
            training_set_years = row['training_length']
            label_size = row['#_per_label']
            year_test = row['year']
            
            group_count += 1
            print(f"{group_count} / {total_count} : {datetime.datetime.now().strftime('%H:%M:%S')}, {fos_name}, {training_set_years}, {label_size}, {year_test}")

            # Select the top label_size rows - the data is already shuffled when df is loaded
            # first, get data for the relevant years,
            # Second, group them by year / isNewFOS and select top n rows from true/false of each year
            Data_train = \
                df.loc[(df['year'] < year_test) & (df['year'] >= year_test - training_set_years)]\
                .groupby(['year', targetLabel]).head(label_size).reset_index(drop=True)
            training_frame = h2o.H2OFrame(Data_train)

            Data_test = \
                df.loc[df['year'] == year_test]\
                    .groupby(['year', targetLabel]).head(label_size).reset_index(drop=True)
            leaderboard_frame = h2o.H2OFrame(Data_test)

            # Run AutoML
            aml = H2OAutoML(max_models = num_models, seed = 1, nfolds=10, verbosity=None, include_algos=["GLM"])
            aml.train(x = properties, y = targetLabel, training_frame=training_frame, leaderboard_frame=leaderboard_frame)

            intercepts = list(aml.leader.coef().values())
            outcome = [[fos_name, lev, maintype, year_test, training_set_years, label_size]  + intercepts]

            pd.DataFrame(outcome).to_csv(outfname, sep='\t', index=False, mode='a', header=None)

        # Clean the models
        h2o.remove_all()

    h2o.cluster().shutdown(prompt = False)

if __name__ == "__main__":
    # set variables
    years_to_record = list(range(1990,2021))
    number_of_pegonets_to_search = 500

    # training_set_years_list = [9, 5, 1]
    # label_size_list = [500, 250, 100]
    # years_to_run = range(1999,2020)
    
    training_set_years_list = [9]
    label_size_list = [500]
    years_to_run = [2000, 2005, 2010, 2015]

    targetLabel = 'isNewFOS'
    classification_fname = './Automl_results.csv'


    group_by = [
        'main_type',
        'fos',
        'year']
    info = [
        'seed',
        'age',
        'isNewFOS',
        '#nodes_in_y+1',
        '#edges_in_y+1']
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
    # prepare_pegonet_data(years_to_record, number_of_pegonets_to_search, columns)

    group_by = [
        'year',
        'fos',
        'training_length',
        '#_per_label']
    # 2. Train using n years, and use +1 year as test set
    # classify_egonet_within_fos_automl(classification_fname, group_by, properties, training_set_years_list, targetLabel, label_size_list, years_to_run)

    coeffic_fname = './GLM_coefficients.csv'
    get_intercepts(coeffic_fname, group_by, properties, training_set_years_list, targetLabel, label_size_list, years_to_run)
