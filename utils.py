import sqlite3
import time
from datetime import datetime
import os
from scipy import stats
import itertools
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import glob
from re import sub

DATASETDIR = 'E:/Dataset/MAG_toplevel'
# DATASETDIR = 'E:/Dataset/MAG_test'


# Return the group_by columns of finished
def check_file_exists(fname):
    return os.path.isfile(fname)

def get_df(main_type, fos_name, do_shuffle=True, rows_to_get=0):
    fname = glob.glob(f'{DATASETDIR}/egonets/{fos_name}__*.csv')[0]
    if main_type != None:
        fname = f'{DATASETDIR}/egonets/{main_type}__{fos_name}.csv'
    df = pd.read_csv(fname, sep='\t', encoding='utf-8')
    if rows_to_get > 0:
        dflist = []
        for _, g in df.groupby(['isNewFOS', 'year']):
            dflist.append(g.head(int(rows_to_get/2)))
        df = pd.concat(dflist)
    if do_shuffle ==True:
        df_shuffled = shuffle(df)
        return df_shuffled
    else:
        return df


def measure_time(func, desc, argv, inline=True):
    timestamp = datetime.now()
    if inline == True:
        print(f"{timestamp.strftime('%H:%M:%S')}: {desc} - ", sep=' ', end='', flush=True)
    else:
        print(f"{timestamp.strftime('%H:%M:%S')}: {desc} - ")
    result = func(*argv)
    print(f"done in {(datetime.now() - timestamp).total_seconds()} seconds.")
    return result

def connect(main_type, fos_name):
    db = f'{DATASETDIR}/mag-{main_type}__{fos_name}.db'
    conn = sqlite3.connect(db)
    return conn

def close(conn):
    conn.close()
    return None

def get_egonet_fname(main_type, fos_name):
    return f'{DATASETDIR}/egonets/{main_type}__{fos_name}.csv'

def get_fos_list():
    return [f.replace('mag-','').replace('.db','').split('__') for f in os.listdir(DATASETDIR) if 'mag-' in f]

def update_fos_level_maintype(fname):
    fos_list = get_fos_list()
    fos_level_mt = [[i, j.split('_',1)[0], j.split('_',1)[1] if '_' in j else None] for i, j in fos_list]
    with open(fname,'r') as f:
        text = f.read()
    
    for fos,level,mt in fos_level_mt:
        text = sub(f"\n{fos}\t", f"\n{fos}\t{level}\t{mt}\t", text)

    with open(fname,'w') as f:
        f.write(text)
    
def get_level_mt(fos_name):
    st = glob.glob(f'{DATASETDIR}/egonets/{fos_name}__*.csv')[0].split('/')[-1].split('__',1)[1]
    lv, mt = st.split('_',1) if '_' in st else [st, None]
    return lv, mt


def get_fos_list_from_egonet():
    return [f.replace('.csv','').split('__') for f in os.listdir(f'{DATASETDIR}/egonets')]

def get_fos_pair_iter(both=True):
    l = list(itertools.combinations(get_fos_list(),2))
    if both==True:
        l2 = [(i,j) for j,i in l]
        l.extend(l2)
    return len(l), l

def check_significance(df, label, group_by, properties):
    results = []
    pvalues = []
    for k, group in df.groupby(group_by):
        a, b = [arr[properties] for _,arr in group.groupby(label)]
        result, pvalue = stats.ttest_ind(a,b,equal_var=False)
        results.append(list(k) + list(result))
        pvalues.append(list(k) + list(pvalue))
    return results, pvalues

def check_significance_anova(df, label, group_by, properties):
    outcome = []
    for k, group in df.groupby(group_by):
        a = [arr[properties] for _,arr in group.groupby(label)]
        result, pvalue = stats.f_oneway(*a)
        outcome.append(list(k) + list(result) + list(pvalue))
    df = pd.DataFrame(outcome, columns=group_by+[p+'_result' for p in properties]+[p+'_pvalue' for p in properties])

    return df


def avg_shortest_path(graph):
    if nx.is_connected(graph):
        return nx.average_shortest_path_length(graph, weight='distance')
    else:
        cc_list = [nodes for nodes in nx.connected_components(graph)]
        return sum([nx.average_shortest_path_length(graph.subgraph(nodes)) for nodes in cc_list]) / float(len(cc_list))

def getstats(subgraph, G, nodes):
    return [
        len(subgraph.nodes()), 
        len(subgraph.edges()),
        sum([d for _,d in subgraph.degree()]) / float(len(subgraph.nodes())),
        sum([d for _,d in subgraph.degree(weight='weight')]) / float(len(subgraph.nodes())),
        sum(nx.get_node_attributes(subgraph, 'age').values()) / float(len(subgraph.nodes())),
        sum(nx.get_edge_attributes(subgraph, 'weight').values()) / float(len(subgraph.edges())),
        avg_shortest_path(subgraph),
        np.average([v for _,v in nx.pagerank(subgraph).items()]),
        np.average([v for _,v in nx.degree_centrality(subgraph).items()]),
        np.average([v for _,v in nx.betweenness_centrality(subgraph, weight='weight').items()]),
        len(subgraph.edges)/len(G.edges(nodes)),
        nx.density(subgraph),
        nx.transitivity(subgraph),
        nx.average_clustering(subgraph, weight='weight'),
        sum(nx.triangles(subgraph).values()) / float(len(subgraph.nodes()))
        ]

def is_finished(outfname, iteration_cols, target_group):
    groups = [i for i,_ in pd.read_csv(outfname, sep='\t', encoding='utf-8').groupby(iteration_cols)]
    return target_group in groups

    
def get_iterations_from_egonet(outfname, iteration_cols, training_set_years_list, label_size_list, years_to_run):
    total_iterations = (
        (year, fos_name, training_set_years, label_size)
        for year in years_to_run
        for fos_name, _ in get_fos_list_from_egonet()
        for training_set_years in training_set_years_list
        for label_size in label_size_list
    )
    finished_iterations = (i for i,_ in pd.read_csv(outfname, sep='\t', encoding='utf-8').groupby(iteration_cols))
    unfinished_iterations = set(total_iterations).difference(finished_iterations)

    unfinished_df = pd.DataFrame(unfinished_iterations, columns=iteration_cols)
    unfinished_groups = unfinished_df.groupby(['fos'])

    return len(unfinished_df), iter(unfinished_groups)
    

#This one is made for the SDG classifier - trying for the 'partial_fit'. This didn't really work out.
def getAccForClassifier(c_type, c, X_train, y_train, X_test, y_test, epoch=-1, train=True):
    if len(set(y_train)) == 1:
        print("Training set too small and contain only one label, skipping")
    elif len(set(y_test)) == 1:
        print("Test set too small and contain only one label, skipping")
    else:
        # Train the dataset if necessary
        if train==True:
            if c_type == 0: # c_cold, just run fit
                c.fit(X_train, y_train)
            elif c_type == 1: # c_warm, just run fit
                c.fit(X_train, y_train)
            elif c_type == 2: # c_partial, run partial_fit for given number of epochs
                for _ in range(epoch):
                    c.partial_fit(X_train, y_train, classes=np.unique(y_train))
        # y_pred_proba = c.predict_proba(X_test)[:, -1]

        acc = c.score(X_test, y_test)
        y_pred = c.predict(X_test)
        ((tn, fp),(fn,tp)) = confusion_matrix(y_test, y_pred)/len(y_pred)
        
        f1 = tp / (tp + 0.5 * (fp + fn))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return c, [acc, prec, rec, f1]
    return c, []

# # This is accuracy classifier without transfer learning in mind.
def getAcc(c, X_train, y_train, X_test, y_test):
    c.fit(X_train, y_train)
    y_pred_proba = c.predict_proba(X_test)[:, -1]

    acc = c.score(X_test, y_test)
    y_pred = c.predict(X_test)
    ((tn, fp),(fn,tp)) = confusion_matrix(y_test, y_pred)/len(y_pred)

    auc = roc_auc_score(y_test, y_pred_proba) 
    f1 = tp / (tp + 0.5 * (fp + fn))
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    return [auc, acc, prec, rec, f1]

def standardize(X_train, X_test):
    m = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    train_st = (X_train - m) / std
    test_st =  (X_test - m) / std

    return train_st, test_st

def get_graph(conn, years_to_record):
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