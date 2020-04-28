import numpy as np
import pandas as pd

train_transaction = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')
test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')

#merge datasets
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

train_transaction.head()

#id_01 - id_11 are continuous variables, id_12-id_38 are categorial variables
train_identity.head()

train_identity.shape

#isfraud, transactionamt, productcd, 
train.shape

test.shape

#del train_identity, train_transaction, test_identity, test_transaction

one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
one_value_cols == one_value_cols_test



#that 1 col with unique value - maybe all unique value or maybe all missing nan value

#print(f'There are {len(one_value_cols)} columns in train dataset with one unique value.')
#print(f'There are {len(one_value_cols_test)} columns in test dataset with one unique value.')

import matplotlib.pyplot as plt
% matplotlib inline

train['isFraud'].value_counts().plot(kind='bar', color = ['orange', 'red'])
plt.title('Distribution of target variable')

train['isFraud'].value_counts()

#transactiondt feature is a timedelta from a given reference datetime, #not an actual timestamp. 
#train and test transaction dates do not overlap-split by time, there is #a light gap in between, impacy the cross validation technique
#so we should use time based split for validation

plt.hist(train['TransactionDT'], label = 'train', color = 'blue')
plt.hist(test['TransactionDT'],label = 'test', color = 'orange')
plt.legend()
plt.title('Distribution of transaction dates')

train['TransactionAmt'].apply(np.log).plot(kind = 'hist', bins = 100, figsize = (15,5), color = 'green', title = 'Distribution of Log Transaction Amt')
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 6))
train.loc[train['isFraud'] == 1] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Fraud',
          xlim=(-3, 10),
         ax= ax1)
train.loc[train['isFraud'] == 0] \
    ['TransactionAmt'].apply(np.log) \
    .plot(kind='hist',
          bins=100,
          title='Log Transaction Amt - Not Fraud',
          xlim=(-3, 10),
         ax=ax2)
train.loc[train['isFraud'] == 1] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Fraud',
         ax= ax3)
train.loc[train['isFraud'] == 0] \
    ['TransactionAmt'] \
    .plot(kind='hist',
          bins=100,
          title='Transaction Amt - Not Fraud',
         ax=ax4)
plt.show()


print('Mean transaction amt for fraud is {:.4f}'.format(train_transaction.loc[train_transaction['isFraud'] == 1]['TransactionAmt'].mean()))
print('Mean transaction amt for non-fraud is {:.4f}'.format(train_transaction.loc[train_transaction['isFraud'] == 0]['TransactionAmt'].mean()))

train.groupby('ProductCD') \
    ['TransactionID'].count() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(15, 3),
         title='Count of Observations by ProductCD', 
         color = ['grey','purple','red','blue','yellow'])
plt.show()


train.groupby('ProductCD')['isFraud'] \
    .mean() \
    .sort_index() \
    .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD',
         color = ['grey','purple','red','blue','yellow'])
plt.show()


import matplotlib.pylab as plt
import seaborn as sns
import warnings


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['card4','isFraud']].groupby(['card4']).mean().plot.bar(ax=ax[0],color = 'orange')
ax[0].set_title('isFraud vs card4')
sns.countplot('card4',hue='isFraud',data=train,ax=ax[1]
             )
ax[1].set_title('card4:Fraud vs non-Fraud')
plt.show()



f,ax=plt.subplots(1,2,figsize=(18,8))
train[['card6','isFraud']].groupby(['card6']).mean().plot.bar(ax=ax[0],color = 'yellow')
ax[0].set_title('isFraud vs card6')
sns.countplot('card6',hue='isFraud',data=train,ax=ax[1]
             )
ax[1].set_title('card6:Fraud vs non-Fraud')
plt.show()

train['dist1'].plot(kind='hist',
                                bins=5000,
                                figsize=(15, 2),
                                title='dist1 distribution',
                                logx=True)
plt.show()
train['dist2'].plot(kind='hist',
                                bins=5000,
                                figsize=(15, 2),
                                title='dist2 distribution',
                                logx=True)
plt.show()




#Possibly this could be the distance of the transaction vs. the card owner's home/work address
#further distance for fraud cases 
print('Mean dist1 amt for fraud is {:.4f}'.format(train.loc[train['isFraud'] == 1]['dist1'].mean()))
print('Mean dist1 amt for non-fraud is {:.4f}'.format(train.loc[train['isFraud'] == 0]['dist1'].mean()))
print('Mean dist2 amt for fraud is {:.4f}'.format(train.loc[train['isFraud'] == 1]['dist2'].mean()))
print('Mean dist2 amt for non-fraud is {:.4f}'.format(train.loc[train['isFraud'] == 0]['dist2'].mean()))



train.groupby('DeviceType') \
    .mean()['isFraud'] \
    .sort_values() \
    .plot(kind='barh',
          figsize=(15, 5),
          title='Percentage of Fraud by Device Type', color = ['blue','red'])
plt.show()


#fraud cases's email association
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
a = train[train['isFraud']==1].groupby('P_emaildomain')['P_emaildomain'].count().sort_values(ascending=False)#.plot(kind='barh', ax=ax1, title='Count of P_emaildomain fraud')
print(a[0:6])
b= train[train['isFraud']==0].groupby('P_emaildomain')['P_emaildomain'].count().sort_values(ascending=False)#.plot(kind='barh', ax=ax2, title='Count of P_emaildomain non-fraud')
print(b[0:6])
#train_transaction_fr.groupby('R_emaildomain')['R_emaildomain'].count()#.plot(kind='barh', ax=ax3, title='Count of R_emaildomain fraud')
#train_transaction_nofr.groupby('R_emaildomain')['R_emaildomain'].count()#.plot(kind='barh', ax=ax4, title='Count of R_emaildomain non-fraud')
#plt.show()

train.groupby('DeviceInfo') \
    .count()['TransactionID'] \
    .sort_values(ascending=False) \
    .head(20) \
    .plot(kind='barh', figsize=(15, 5), title='Top 20 Devices in Train')
plt.show()

import gc
nans_df = train.isna()
nans_groups={}
i_cols = ['V'+str(i) for i in range(1,340)]
for col in train.columns:
    cur_group = nans_df[col].sum()
    try:
        nans_groups[cur_group].append(col)
    except:
        nans_groups[cur_group]=[col]
del nans_df; x=gc.collect()

for k,v in nans_groups.items():
    print('####### NAN count =',k)
    print(v)
    
Vc = ['dayr','isFraud','TransactionAmt','card1','addr1','D1n','D11n']
Vs = nans_groups[279287]
Vtitle = 'V1 - V11, D11'

def make_plots(Vs):
    col = 4
    row = len(Vs)//4+1
    plt.figure(figsize=(20,row*5))
    idx = train[~train[Vs[0]].isna()].index
    for i,v in enumerate(Vs):
        plt.subplot(row,col,i+1)
        n = train[v].nunique()
        x = np.sum(train.loc[idx,v]!=train.loc[idx,v].astype(int))
        y = np.round(100*np.sum(train[v].isna())/len(train),2)
        t = 'int'
        if x!=0: t = 'float'
        plt.title(v+' has '+str(n)+' '+t+' and '+str(y)+'% nan')
        plt.yticks([])
        h = plt.hist(train.loc[idx,v],bins=100)
        if len(h[0])>1: plt.ylim((0,np.sort(h[0])[-2]))
    plt.show()
make_plots(Vs)

def make_corr(Vs,Vtitle=''):
    cols = ['TransactionDT'] + Vs
    plt.figure(figsize=(15,15))
    sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
    if Vtitle!='': plt.title(Vtitle,fontsize=14)
    else: plt.title(Vs[0]+' - '+Vs[-1],fontsize=14)
    plt.show()
make_corr(Vs,Vtitle)

grps = [[1],[2,3],[4,5],[6,7],[8,9],[10,11]]
def reduce_group(grps,c='V'):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = train[c+str(gg)].nunique()
            if n>mx:
                mx = n
                vx = gg
            #print(str(gg)+'-'+str(n),', ',end='')
        use.append(vx)
        #print()
    print('Use these',use)
reduce_group(grps)


Vs = nans_groups[76073]
make_plots(Vs)
make_corr(Vs)

grps = [[12,13],[14],[15,16,17,18,21,22,31,32,33,34],[19,20],[23,24],[25,26],[27,28],[29,30]]
reduce_group(grps)


Vs = nans_groups[168969]
make_plots(Vs)
make_corr(Vs)

grps = [[35,36],[37,38],[39,40,42,43,50,51,52],[41],[44,45],[46,47],[48,49]]
reduce_group(grps)

Vs = nans_groups[77096]
make_plots(Vs)
make_corr(Vs)

grps = [[53,54],[55,56],[57,58,59,60,63,64,71,72,73,74],[61,62],[65],[66,67],[68],[69,70]]
reduce_group(grps)

Vs = nans_groups[89164]
make_plots(Vs)
make_corr(Vs)

grps = [[75,76],[77,78],[79,80,81,84,85,92,93,94],[82,83],[86,87],[88],[89],[90,91]]
reduce_group(grps)

Vs = nans_groups[314]
make_corr(Vs)

Vs = ['V'+str(x) for x in range(95,107)]
make_plots(Vs)
make_corr(Vs)

grps = [[95,96,97,101,102,103,105,106],[98],[99,100],[104]]
reduce_group(grps)

Vs = ['V'+str(x) for x in range(107,124)]
make_plots(Vs)
make_corr(Vs)

grps = [[107],[108,109,110,114],[111,112,113],[115,116],[117,118,119],[120,122],[121],[123]]
reduce_group(grps)

Vs = ['V'+str(x) for x in range(124,138)]
make_plots(Vs)
make_corr(Vs)

grps = [[124,125],[126,127,128,132,133,134],[129],[130,131],[135,136,137]]
reduce_group(grps)

Vs = nans_groups[508595]
make_plots(Vs)
make_corr(Vs)

grps = [[138],[139,140],[141,142],[146,147],[148,149,153,154,156,157,158],[161,162,163]]
reduce_group(grps)

grps = [[143,164,165],[144,145,150,151,152,159,160],[166]]
reduce_group(grps)

grps = [[167,168,177,178,179],[172,176],[173],[181,182,183]]
reduce_group(grps)

grps = [[186,187,190,191,192,193,196,199],[202,203,204,211,212,213],[205,206],[207],[214,215,216]]
reduce_group(grps)

grps = [[169],[170,171,200,201],[174,175],[180],[184,185],[188,189],[194,195,197,198],[208,210],[209]]
reduce_group(grps)

grps = [[217,218,219,231,232,233,236,237],[223],[224,225],[226],[228],[229,230],[235]]
reduce_group(grps)

grps = [[240,241],[242,243,244,258],[246,257],[247,248,249,253,254],[252],[260],[261,262]]
reduce_group(grps)

grps = [[263,265,264],[266,269],[267,268],[273,274,275],[276,277,278]]
reduce_group(grps)

grps = [[220],[221,222,227,245,255,256,259],[234],[238,239],[250,251],[270,271,272]]
reduce_group(grps)

grps = [[279,280,293,294,295,298,299],[284],[285,287],[286],[290,291,292],[297]]
reduce_group(grps)

grps = [[302,303,304],[305],[306,307,308,316,317,318],[309,311],[310,312],[319,320,321]]
reduce_group(grps)

grps = [[281],[282,283],[288,289],[296],[300,301],[313,314,315]]
reduce_group(grps)

grps = [[322,323,324,326,327,328,329,330,331,332,333],[325],[334,335,336],[337,338,339]]
reduce_group(grps)

v =  [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]
v += [96, 98, 99, 104]
v += [107, 108, 111, 115, 117, 120, 121, 123]
v += [124, 127, 129, 130, 136]
v += [138, 139, 142, 147, 156, 162]
v += [165, 160, 166]
v += [178, 176, 173, 182]
v += [187, 203, 205, 207, 215]
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
v += [218, 223, 224, 226, 228, 229, 235]
v += [240, 258, 257, 253, 252, 260, 261]
v += [264, 266, 267, 274, 277]
v += [220, 221, 234, 238, 250, 271]
v += [294, 284, 285, 286, 291, 297]
v += [303, 305, 307, 309, 310, 320]
v += [281, 283, 289, 296, 301, 314]
v += [332, 325, 335, 338]

print('Reduced set has',len(v),'columns')

str("v")+str(v[0])

#get list of v we should keep to be used to delete rest of v we do not need
updated_v_col = []
for i in v:
    updated_v_col.append(str('V')+str(i))
    
cols = ['TransactionDT'] + ['V'+str(x) for x in v]
train2 = train[cols].sample(frac=0.2)
plt.figure(figsize=(15,15))
sns.heatmap(train2[cols].corr(), cmap='RdBu_r', annot=False, center=0.0)
plt.title('V1-V339 REDUCED',fontsize=14)
plt.show()

cols = ['TransactionDT'] + ['V'+str(x) for x in range(1,340)]
train2 = train[cols].sample(frac=0.2)
plt.figure(figsize=(15,15))
sns.heatmap(train2[cols].corr(), cmap='RdBu_r', annot=False, center=0.0)
plt.title('V1-V339 ALL',fontsize=14)
plt.show()

cols = ['V'+str(x) for x in v]
pca_v = train[cols]

pca_v.shape

from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca_v.columns

features = []
for item in pca_v.columns:
    features.append(item)
    
#standardize reduced v group 

from sklearn.preprocessing import StandardScaler

reduced_v = train.loc[:, features].values
y = train.loc[:, ['isFraud']].values

# Standardizing the reduced v features
x = StandardScaler().fit_transform(reduced_v)



type(x)

import numpy.ma as ma
x = np.where(np.isnan(x), ma.array(x, mask=np.isnan(x)).mean(axis=0), x)    

#perform PCA on reduced - V group 
from sklearn.decomposition import PCA
pca = PCA(0.90)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)

#50 principal components generated 

principalDf 

len(v)

len(cols)

for item in train.columns:
    print(item)
    
useful_features = ['TransactionDT','TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'dist2','P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15','M1', 'M2', 'M3',
                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_01','id_02','id_03','id_04',
                  'id_05','id_06','id_07','id_08','id_09','id_10','id_11','id_12',
                  'id_13','id_14','id_15','id_16','id_17','id_18','id_19','id_20',
                  'id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28',
                  'id_29','id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37',
                  'id_38','DeviceType','DeviceInfo']




cols_to_drop = [col for col in train.columns if col not in useful_features]

print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

train.shape

type(principalDf)

simple_train_plusreudced_v = pd.concat([train,principalDf ], axis=1)

simple_train_plusreudced_v.shape

simple_train_plusreudced_v.fillna(-999, inplace=True)

simple_train_plusreudced_v['Transaction_day_of_week'] = np.floor((simple_train_plusreudced_v['TransactionDT'] / (3600 * 24) - 1) % 7)
#simple_train_plusreudced_v['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
simple_train_plusreudced_v['Transaction_hour'] = np.floor(simple_train_plusreudced_v['TransactionDT'] / 3600) % 24
#simple_train_plusreudced_v['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



cat_cols = ['DeviceType', 'ProductCD', 'card4', 'card6','Transaction_day_of_week','Transaction_hour']

dummies_df = pd.get_dummies(simple_train_plusreudced_v[cat_cols])

dummies_df.shape

dummies_df.columns

simple_train_plusreudced_v = simple_train_plusreudced_v.drop(['DeviceType',
                                                              'ProductCD',
                                                              'card4','card6',
                                                              'Transaction_day_of_week',
                                                             'Transaction_hour'], axis=1)

concated_2_df = pd.concat([simple_train_plusreudced_v, dummies_df ], axis=1)

concated_2_df.head()

concated_2_df = concated_2_df.drop(['TransactionDT'], axis=1)

concated_2_df.columns

y_train = concated_2_df.isFraud

y_train.shape

x_train = concated_2_df.drop(['isFraud'], axis = 1)

x_train.shape

x_train_2 = x_train[0:390540]

y_train_2 = y_train[0:390540]

x_train_3 = x_train[390540:]
y_train_3 = y_train[390540:]

params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }



import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
%matplotlib inline

x_train = x_train.drop(['P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 
                        'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 
                        'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 
                        'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 
                        'id_37', 'id_38', 'DeviceInfo'], axis = 1)

folds = TimeSeriesSplit(n_splits=5)

aucs = list()
#feature_importances = pd.DataFrame()
#feature_importances['feature'] = X.columns

#training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(x_train, y_train)):
#    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])
    val_data = lgb.Dataset(x_train.iloc[test_idx], label=y_train.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=5)
    
#    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    #print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
#print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)



from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from sklearn.metrics import make_scorer

import time
def objective(params):
    time1 = time.time()
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 7
    count=1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    tss = TimeSeriesSplit(n_splits=FOLDS)
    y_preds = np.zeros(sample_submission.shape[0])
    y_oof = np.zeros(x_train.shape[0])
    score_mean = 0
    for tr_idx, val_idx in tss.split(x_train, y_train):
        clf = xgb.XGBClassifier(
            n_estimators=600, random_state=4, verbose=True, 
            tree_method='gpu_hist', 
            **params
        )

        x_tr, x_vl = x_train.iloc[tr_idx, :], x_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(x_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, x_vl, y_vl)
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1
    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del x_tr, x_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)


space = {
    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model 
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    'max_depth': hp.quniform('max_depth', 7, 23, 1),
    
    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    
    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing 
    # features might not make much sense.
    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
    
    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    
    # colsample_bytree: Similar to max_features in GBM. Denotes the 
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
    
    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the 
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    'gamma': hp.uniform('gamma', 0.01, .7),
    
    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number 
    # of leaves will improve accuracy, but will also lead to overfitting.
    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
    
    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf. 
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
    
    # subsample: represents a fraction of the rows (observations) to be 
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend 
    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
    
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in 
    # the case of bagging). Smaller fractions reduce overfitting.
    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
    
    # randomly bag or subsample training data.
    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
    
    # bagging_fraction and bagging_freq: enables bagging (subsampling) 
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.



print("BEST PARAMS: ", best_params)

best_params['max_depth'] = int(best_params['max_depth'])


clf = xgb.XGBClassifier(
    n_estimators=300,
    **best_params,
    tree_method='gpu_hist'
)

clf.fit(x_train, y_train)

y_preds = clf.predict_proba(X_test)[:,1] 

feature_important = clf.get_booster().get_score(importance_type="weight")
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

# Top 10 features
data.head(20)