import pandas as pd
import scipy
from scipy.stats import ttest_ind
import scipy.stats as ss
import numpy as np


# # 
# # is there difference between l/t combinations?
# # 
# df = pd.read_csv('Automl_results.csv', encoding='utf-8')
# grouped_cols = ['training_length', '#_per_label']
# out = []
# for combi, df_group in df.groupby(grouped_cols):
#     df_group = df_group.loc[df_group['model_rank']==1]
#     for find in ['auc','logloss','aucpr','mean_per_class_error','rmse','mse']:
#         samples = [group for label, group in df_group.groupby(['model'])[find]]
#         f_val, p_val = ss.f_oneway(*samples)
#         out.append(combi + (find, f_val, p_val))
# df_out = pd.DataFrame(out, columns=grouped_cols+['measure','f_val','p_val'])
# print(df_out)


# # 
# # is there difference between l/t combinations?
# # 
# df = pd.read_csv('Prediction_basedon_GLM_coeff.csv', encoding='utf-8')
# grouped_cols = ['#features', 'level', 'same_level']
# out = []
# for combi, df_group in df.groupby(grouped_cols):
#     samples = [group for label, group in df_group.groupby(['l'])['F1']]
#     f_val, p_val = ss.f_oneway(*samples)
#     out.append(combi + (f_val, p_val))
# df_out = pd.DataFrame(out, columns=grouped_cols+['f_val','p_val'])
# print(df_out)


# 
# is there difference between 3 property values among different fos?
# 
# 1. get average properties for 310 fos
# 2. run anova for all 310
# 3. run t-test between good/bad ones
# import utils
# out = []
# features = ['#nodes', '#edges', 'avg_degree', 'avg_degree_freqWeighted', 'avg_n_age', 'avg_e_weight', 'avg_shortest_path', 'avg_pagerank', 'avg_degree_centrality', 'avg_betweeness_centrality', 'cohesion', 'density', 'transitivity', 'average_clustering', 'avg_triangles']
# badfos = ['endocrinology', 'internal medicine', 'surgery', 'medicine', 'cardiology', 'keynesian economics', 'biology', 'classical economics', 'neoclassical economics', 'artificial intelligence', 'polymer science', 'composite material', 'computer vision', 'pathology', 'computer science', 'gastroenterology', 'biochemical engineering', 'astrobiology', 'chemistry', 'earth science']
# lall = []; lgood=[]; lbad=[]
# for fos_name,suffix in utils.get_fos_list_from_egonet():
#     if fos_name == "ceramic materials":
#         continue # don't work on ceramic materials
#     df = utils.get_df(None, fos_name)[features]
#     avg = df.mean(axis=0)
#     lall.append(avg)
#     if fos_name in badfos:
#         lbad.append(avg)
#     else:
#         lgood.append(avg)

# df_all = pd.concat(lall,axis=1)
# df_good = pd.concat(lgood,axis=1)
# df_bad = pd.concat(lbad,axis=1)


# # is there difference between bad/good ones
# bad1 = ['endocrinology', 'internal medicine', 'surgery', 'medicine', 'cardiology']
# bad2 = ['keynesian economics', 'classical economics', 'neoclassical economics']
# good = ['atomic physics', 'photochemistry', 'condensed matter physics', 'thermodynamics', 'public relations', 'economics', 'pedagogy', 'geotechnical engineering', 'waste management', 'environmental science']
# features = ['#nodes', '#edges', 'avg_degree', 'avg_degree_freqWeighted', 'avg_n_age', 'avg_e_weight', 'avg_shortest_path', 'avg_pagerank', 'avg_degree_centrality', 'avg_betweeness_centrality', 'cohesion', 'density', 'transitivity', 'average_clustering', 'avg_triangles']

# l1 = []
# l2 = []
# lg = []
# import utils
# for fos_name,suffix in utils.get_fos_list_from_egonet():
#     if fos_name == "ceramic materials":
#         continue # don't work on ceramic materials
#     df = utils.get_df(None, fos_name)[features]
#     if fos_name in bad1:
#         l1.append(df.mean(axis=0))
#     elif fos_name in bad2:
#         l2.append(df.mean(axis=0))
#     elif fos_name in good:
#         lg.append(df.mean(axis=0))

# df1 = pd.concat(l1,axis=1)
# df2 = pd.concat(l2,axis=1)
# dfg = pd.concat(lg,axis=1)

# df1.to_csv('./bad1.out')
# df2.to_csv('./bad2.out')
# dfg.to_csv('./good.out')


bad1 = ['endocrinology', 'internal medicine', 'surgery', 'medicine', 'cardiology']
bad2 = ['keynesian economics', 'classical economics', 'neoclassical economics']
good = ['atomic physics', 'photochemistry', 'condensed matter physics', 'thermodynamics', 'public relations', 'economics', 'pedagogy', 'geotechnical engineering', 'waste management', 'environmental science']
features = ['auc','logloss','aucpr','mean_per_class_error','rmse','mse']
import utils
df = pd.read_csv('Automl_results_GLM_9_500.csv', encoding='utf-8')
print(df.loc[df['fos'].isin(bad1)][features].mean(axis=0))
print(df.loc[df['fos'].isin(bad2)][features].mean(axis=0))
print(df.loc[df['fos'].isin(good)][features].mean(axis=0))