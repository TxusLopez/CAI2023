import numpy as np
import pandas as pd
from scipy.stats import t
from functools import reduce
from river import preprocessing
from river import linear_model
from river import optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from river import compose
from river import metrics
from river import stats as riv_stats
from scipy import stats as sc_stats
from river import stream
from river.datasets import water_flow

random_state = np.random.RandomState(42)

####################################################### FUNCTIONS #######################################################

def generateRegressionDataSet(intercept=[1,0], domain=range(0,500), noise=[0 for x in range(0,500)], columns=['x', 'y']):
  data = []
  for x, single_noise in zip(domain, noise):
    data.append([x, x*intercept[0]+intercept[1]+ single_noise])
  return pd.DataFrame(data, columns=columns)

def mergeDataSetsSudden(dataset1, dataset2):
  res = []
  for data in dataset1.values.tolist():
    res.append(data)
  for data in dataset2.values.tolist():
    res.append(data)
  return pd.DataFrame(data=res, columns=dataset1.columns)

def generateRegressionDataSetWithSuddenDrift(intercept, domain=[range(0,500), range(500,1000)], noise=[[0 for x in range(0,500)], [0 for x in range(0,500)]], columns=['x', 'y']):
  datasets = []
  for d in zip(domain, noise, intercept):
    dataset = generateRegressionDataSet(intercept=d[2], domain=d[0], noise=d[1], columns=columns)
    datasets.append(dataset)
  return reduce(mergeDataSetsSudden, datasets)

def normalizeData(dataset):
  return (dataset-dataset.mean())/dataset.std()

def make_model_synth(alpha):

  scale = preprocessing.StandardScaler()

  learn = linear_model.LinearRegression(
    intercept_lr=0,
    optimizer=optim.Adam(),
    loss=optim.losses.Quantile(alpha=alpha)
  )

  model = scale | learn

  return model

def make_model_real(alpha):

  learn = linear_model.LinearRegression(
    intercept_lr=0.1,
    intercept_init=0.01,
    optimizer=optim.Adam(),
    loss=optim.losses.Quantile(alpha=alpha)
  )

  model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', learn)
  )

  return model

def get_ordinal_date(x):
  return {'ordinal_date': x['Time'].toordinal()}


####################################################### MAIN #######################################################

n_samples=500

#########################
# Synthetic Dataset: SUDDEN DRIFT
#########################

#Data generation
data_synth = normalizeData(generateRegressionDataSetWithSuddenDrift(intercept=[[0,10], [0,30]], noise=[random_state.normal(0,2, 500), random_state.normal(0,2, 500)]))

#Stream Learning process

sorted_dataset = data_synth.sort_values(by=['x'])
X = list(map(lambda x: {'x': x}, sorted_dataset['x']))
Y = sorted_dataset['y'].values.tolist()

metric_synth = metrics.MAE()
maes_synth=[]
dif_interval=[]

models_synth = {
    'lower': make_model_synth(alpha=0.05),
    'center': make_model_synth(alpha=0.5),
    'upper': make_model_synth(alpha=0.95)
}

y_trues_synth = []
y_preds_synth = {
    'lower': [],
    'center': [],
    'upper': []
}

for x, y in zip(X, Y):
  y_trues_synth.append(y)

  for name, model in models_synth.items():
    y_preds_synth[name].append(model.predict_one(x))
    model.learn_one(x, y)

  dif_interval.append(y_preds_synth['upper'][-1]-y_preds_synth['lower'][-1])

  # Update the error metric
  metric_synth.update(y, y_preds_synth['center'][-1])
  maes_synth.append(metric_synth.get())

# Plot the results
plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(alpha=0.75)
ax.set_xlabel(r'$Time\ steps$')
ax.set_ylabel(r'$Sensor\ reading\ values$')
ax.plot(sorted_dataset.index,y_trues_synth, lw=3, color='#2ecc71', alpha=0.8, label=r'$Real\ sensor\ reading$')
ax.plot(sorted_dataset.index,y_preds_synth['center'], lw=3, color='#e74c3c', alpha=0.8, label=r'$Prediction\ sensor\ reading$')
# ax.plot(sorted_dataset.index,dif_interval, lw=3, color='orange', alpha=0.8, label='Diff Prediction interval')
ax.fill_between(sorted_dataset.index,y_preds_synth['lower'], y_preds_synth['upper'], color='#e74c3c', alpha=0.3, label=r'$Prediction\ interval$')
plt.axvline(x=(data_synth.shape[0]/2)-1,linestyle='--',color='k')
ax.grid(False)
# ax.annotate(r'Concept 1',xy=(150,2.0), fontsize=12)
# ax.annotate(r'Concept 2',xy=(750,-1.0), fontsize=12)
ax.legend(loc='upper left',fancybox=True,framealpha=1.0,frameon=True)
ax.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel(r'$MAE$', color='tab:blue')  # we already handled the x-label with ax1
ax2.plot(sorted_dataset.index, maes_synth, color='tab:blue', lw=3,alpha=0.8, label=r'$MAE\ evolution$')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='lower right', fancybox=True, framealpha=0.3)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

plt.savefig('../images/sudden.png')

################### Check correlation between uncertainty and drift through MAE and the diff of the interval range

ts_corr=np.arange(475,525,1)

df_corr=pd.DataFrame()
df_corr['dif_interval']=dif_interval[475:525]
df_corr['mae']=maes_synth[475:525]
df_corr=df_corr.set_index(ts_corr)

#Standardization with z-score
# df_corr['dif_interval'] = sc_stats.zscore(df_corr['dif_interval'])
# df_corr['mae'] = sc_stats.zscore(df_corr['mae'])

print('Pearson coefficient: ',df_corr.corr('pearson'))
# print('Spearman coefficient: ',df_corr.corr('spearman'))

# df_corr.plot()

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(alpha=0.75)
ax.set_xlabel(r'$Time\ steps$')
ax.set_ylabel(r'$Prediction\ interval\ values$')
ax.plot(df_corr.index,df_corr['dif_interval'].values, lw=3, color='#e74c3c', alpha=0.8, label=r'$(UQ-PQ)\ evolution$')

# plt.axvline(x=499,linestyle='--',color='k')
ax.grid(False)
# ax.annotate('Concept 1',xy=(150,2.0), fontsize=12)
# ax.annotate('Concept 2',xy=(750,-1.0), fontsize=12)
ax.legend(loc='lower right')
ax.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel(r'$MAE$', color='tab:blue')  # we already handled the x-label with ax1
ax2.plot(df_corr.index,df_corr['mae'].values, lw=3, color='tab:blue', alpha=0.8, label=r'$MAE\ evolution$')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.legend(loc='center right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

plt.savefig('../images/corr.png')
