# _*_ coding: utf-8 _*_

"""
Learning the 1st place solution of 'Corporación Favorita Grocery Sales Forecasting'
competition in Kaggle.

Author: StrongXGP
Date:   2018/11/13
"""

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ====================================================================================== #
# 1. 预处理

print("[INFO] Loading data...")
t0 = time()

# df_train = pd.read_csv(
#     "../data/train.csv", usecols=[1, 2, 3, 4, 5],  # 舍弃id
#     dtype={'onpromotion': bool}, parse_dates=['date'],
#     converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
#     skiprows=range(1, 66458909)
# )
# df_2017 = df_train[df_train['date'] >= '2017-01-01']

df_2017 = pd.read_csv("../data/train-2017.csv", parse_dates=['date'])
df_test = pd.read_csv(
    "../data/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool}, parse_dates=['date']
).set_index(['store_nbr', 'item_nbr', 'date'])
stores = pd.read_csv("../data/stores.csv").set_index("store_nbr")
items = pd.read_csv("../data/items.csv").set_index('item_nbr')

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

encoder = LabelEncoder()
stores['city'] = encoder.fit_transform(stores['city'].values)
stores['state'] = encoder.fit_transform(stores['state'].values)
stores['type'] = encoder.fit_transform(stores['type'].values)
items['family'] = encoder.fit_transform(items['family'].values)

# 每个店铺商品每天的促销情况
promo_2017_train = df_2017.set_index(
    ['store_nbr', 'item_nbr', 'date'])[['onpromotion']].unstack(level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[['onpromotion']].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_train, promo_2017_test
gc.collect()

# 每个店铺商品每天的销量
df_2017 = df_2017.set_index(
    ['store_nbr', 'item_nbr', 'date'])[['unit_sales']].unstack(level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

stores = stores.reindex(df_2017.index.get_level_values(0))
items = items.reindex(df_2017.index.get_level_values(1))

# 每件商品每天的销量
df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()

# 每件商品每天的促销情况
promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()

# 每个种类（的商品）在每个店铺每天的销量
df_2017_store_class = df_2017.reset_index()
df_2017_store_class['class'] = items['class'].values
df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()

# 每个种类（的商品）在每个店铺每天的促销情况
promo_2017_store_class = promo_2017.reset_index()
promo_2017_store_class['class'] = items['class'].values
promo_2017_store_class_index = promo_2017_store_class[['class', 'store_nbr']]
promo_2017_store_class = promo_2017_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()

# ====================================================================================== #

# ====================================================================================== #
# 2. Feature engineering


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_dataset(df, promo_df, t2017, is_train=True, name_prefix=None):
    # 促销天数特征（6个特征）
    X = {
        # 'promo_14_2017': get_timespan(promo_df, t2017, 14, 14).sum(axis=1).values,  # 前14天中每个店铺商品的促销天数
        # 'promo_60_2017': get_timespan(promo_df, t2017, 60, 60).sum(axis=1).values,  # 前60天中每个店铺商品的促销天数
        # 'promo_140_2017': get_timespan(promo_df, t2017, 140, 140).sum(axis=1).values,  # 前140天中每个店铺商品的促销天数
        'promo_3_2017_aft': get_timespan(promo_df, t2017 + timedelta(days=16), 15, 3).sum(
            axis=1).values,  # 后3天中每个店铺商品的促销天数
        'promo_7_2017_aft': get_timespan(promo_df, t2017 + timedelta(days=16), 15, 7).sum(
            axis=1).values,  # 后7天中每个店铺商品的促销天数
        'promo_14_2017_aft': get_timespan(promo_df, t2017 + timedelta(days=16), 15, 14).sum(
            axis=1).values  # 后14天中每个店铺商品的促销天数
    }

    # 促销销量和正常销量特征（24个特征）
    for i in [3, 7, 14, 30, 60, 140]:
        tmp1 = get_timespan(df, t2017, i, i)  # 前i天每个店铺商品的销量
        tmp2 = (get_timespan(promo_df, t2017, i, i) > 0) * 1  # 前i天每个店铺商品的促销情况（是否促销）

        X['has_promo_mean_%s' % i] = (tmp1 * tmp2.replace(0, np.nan)).mean(axis=1).values  # 前i天每个店铺商品平均促销销量
        X['has_promo_mean_%s_decay' % i] = (tmp1 * tmp2.replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(
            axis=1).values  # 前i天每个店铺商品促销销量和（带衰减）

        X['no_promo_mean_%s' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan)).mean(axis=1).values  # 前i天每个店铺商品平均正常销量
        X['no_promo_mean_%s_decay' % i] = (tmp1 * (1 - tmp2).replace(0, np.nan) * np.power(0.9, np.arange(i)[::-1])).sum(
            axis=1).values  # 前i天每个店铺商品正常销量和（带衰减）

    # 销量统计特征（42个特征）
    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)  # 前i天每个店铺商品的销量
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values  # 前i天每个店铺商品销量的平均一阶差分
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values  # 前i天每个店铺商品销量的和（带衰减）
        X['mean_%s' % i] = tmp.mean(axis=1).values  # 前i天每个店铺商品销量的均值
        X['median_%s' % i] = tmp.median(axis=1).values  # 前i天每个店铺商品销量的中位数
        X['min_%s' % i] = tmp.min(axis=1).values  # 前i天每个店铺商品销量的最小值
        X['max_%s' % i] = tmp.max(axis=1).values  # 前i天每个店铺商品销量的最大值
        X['std_%s' % i] = tmp.std(axis=1).values  # 前i天每个店铺商品销量的标准差

    # 销量统计特征2（42个特征）
    for i in [3, 7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017 + timedelta(days=-7), i, i)  # 7天前的前i天每个店铺商品的销量
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values  # 7天前的前i天每个店铺商品销量的平均一阶差分
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(
            axis=1).values  # 7天前的前i天每个店铺商品销量的和（带衰减）
        X['mean_%s_2' % i] = tmp.mean(axis=1).values  # 7天前的前i天每个店铺商品销量的平均值
        X['median_%s_2' % i] = tmp.median(axis=1).values  # 7天前的前i天每个店铺商品销量的中位数
        X['min_%s_2' % i] = tmp.min(axis=1).values  # 7天前的前i天每个店铺商品销量的最小值
        X['max_%s_2' % i] = tmp.max(axis=1).values  # 7天前的前i天每个店铺商品销量的最大值
        X['std_%s_2' % i] = tmp.std(axis=1).values  # 7天前的前i天每个店铺商品销量的标准差

    # 有销量和有促销的天数特征（30个特征）
    for i in [7, 14, 30, 60, 140]:
        tmp = get_timespan(df, t2017, i, i)  # 前i天每个店铺商品的销量
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i天每个店铺商品有销量的天数
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(
            axis=1).values  # 前i天每个店铺商品距离上一次有销量的天数
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(
            axis=1).values  # 前i天每个店铺商品距离第一次有销量的天数

        tmp = get_timespan(promo_df, t2017, i, i)  # 前i天每个店铺商品的促销情况
        X['has_promo_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values  # 前i天每个店铺商品有促销的天数
        X['last_has_promo_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(
            axis=1).values  # 前i天每个店铺商品距离上一次有促销的天数
        X['first_has_promo_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(
            axis=1).values  # 前i天每个店铺商品距离第一次有促销的天数

    # 后15天的促销情况特征（3个特征）
    tmp = get_timespan(promo_df, t2017 + timedelta(days=16), 15, 15)  # 后15天每个店铺商品的促销情况
    X['has_promo_days_in_after_15_days'] = (tmp > 0).sum(axis=1).values  # 后15天每个店铺商品有促销的天数
    X['last_has_promo_day_in_after_15_days'] = ((tmp > 0) * np.arange(1, 16)).max(
        axis=1).values  # 后15天每个店铺商品距离最后一次有促销的天数
    X['first_has_promo_day_in_after_15_days'] = 16 - ((tmp > 0) * np.arange(15, 0, -1)).max(
        axis=1).values  # 后15天每个店铺商品距离最近一次有促销的天数

    # 前15天的销量（15个特征）
    for i in range(1, 16):
        X['day_%s_2017' % i] = get_timespan(df, t2017, i, 1).values.ravel()

    # 前4（20）周每个星期几的平均销量（14个特征）
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df, t2017, 140 - i, 20, freq='7D').mean(axis=1).values

    # 前16天到后15天每天的促销情况（32个特征）
    for i in range(-16, 16):
        X['promo_{}'.format(i)] = promo_df[t2017 + timedelta(days=i)].values.astype(np.uint8)

    X = pd.DataFrame(X)

    if is_train:
        y = df[pd.date_range(t2017, periods=16)].values
        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]

    return X


# 准备训练集
print("[INFO] Preparing training data...")

t2017 = date(2017, 6, 14)
num_days = 6
X_l, y_l = [], []
for i in range(num_days):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(df_2017, promo_2017, t2017 + delta)

    X_tmp2 = prepare_dataset(df_2017_item, promo_2017_item, t2017 + delta, is_train=False, name_prefix='item')
    X_tmp2.index = df_2017_item.index
    X_tmp2 = X_tmp2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

    X_tmp3 = prepare_dataset(df_2017_store_class, promo_2017_store_class, t2017 + delta, is_train=False,
                             name_prefix='store_class')
    X_tmp3.index = df_2017_store_class.index
    X_tmp3 = X_tmp3.reindex(df_2017_store_class_index).reset_index(drop=True)

    X_tmp = pd.concat([X_tmp, X_tmp2, X_tmp3, items.reset_index(), stores.reset_index()], axis=1)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

    del X_tmp, y_tmp, X_tmp2, X_tmp3
    gc.collect()

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)

del X_l, y_l
gc.collect()

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

# 准备验证集
print("[INFO] Preparing validation data...")

X_val, y_val = prepare_dataset(df_2017, promo_2017, date(2017, 7, 26))

X_val2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 7, 26), is_train=False, name_prefix='item')
X_val2.index = df_2017_item.index
X_val2 = X_val2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_val3 = prepare_dataset(
    df_2017_store_class, promo_2017_store_class, date(2017, 7, 26), is_train=False, name_prefix='store_class')
X_val3.index = df_2017_store_class.index
X_val3 = X_val3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_val = pd.concat([X_val, X_val2, X_val3, items.reset_index(), stores.reset_index()], axis=1)

del X_val2, X_val3
gc.collect()

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

# 准备测试集
print("[INFO] Preparing testing data...")

X_test = prepare_dataset(df_2017, promo_2017, date(2017, 8, 16), is_train=False)

X_test2 = prepare_dataset(df_2017_item, promo_2017_item, date(2017, 8, 16), is_train=False, name_prefix='item')
X_test2.index = df_2017_item.index
X_test2 = X_test2.reindex(df_2017.index.get_level_values(1)).reset_index(drop=True)

X_test3 = prepare_dataset(df_2017_store_class, promo_2017_store_class, date(2017, 8, 16), is_train=False, name_prefix='store_class')
X_test3.index = df_2017_store_class.index
X_test3 = X_test3.reindex(df_2017_store_class_index).reset_index(drop=True)

X_test = pd.concat([X_test, X_test2, X_test3, items.reset_index(), stores.reset_index()], axis=1)

del X_test2, X_test3
gc.collect()

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

# ====================================================================================== #

# ====================================================================================== #
# 3. Train a model and predict

print("[INFO] Start training and predicting...")
t0 = time()

params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}

MAX_ROUNDS = 5000
val_pred = []
test_pred = []
cate_vars = []
for i in range(16):
    print('=' * 50)
    print("Step %d" % (i + 1))
    print('=' * 50)

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items['perishable']] * num_days) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        categorical_feature=cate_vars,
        weight=items['perishable'] * 0.25 + 1
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=125,
        verbose_eval=50
    )
    print('\n'.join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance('gain')), key=lambda x: x[1], reverse=True)))
    val_pred.append(
        bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(
        bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

print("Validation mse:", mean_squared_error(y_val, np.array(val_pred).transpose()))

weight = items['perishable'] * 0.25 + 1
err = (y_val - np.array(val_pred).transpose()) ** 2
err = err.sum(axis=1) * weight
err = np.sqrt(err.sum() / weight.sum() / 16)
print("nwrmsle = {}".format(err))

y_val = np.array(val_pred).transpose()
df_preds = pd.DataFrame(
    y_val, index=df_2017.index,
    columns=pd.date_range('2017-07-26', periods=16)
).stack().to_frame('unit_sales')
df_preds.index.set_names(['store_nbr', 'item_nbr', 'date'], inplace=True)
df_preds['unit_sales'] = np.clip(np.expm1(df_preds['unit_sales']), 0, 1000)
df_preds.reset_index().to_csv('lgb_cv.csv', index=False)

# ====================================================================================== #

# ====================================================================================== #
# 4. Make a submission

print("[INFO] Make submission...")
t0 = time()

y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range('2017-08-16', periods=16)
).stack().to_frame('unit_sales')
df_preds.index.set_names(['store_nbr', 'item_nbr', 'date'], inplace=True)

submission = df_test[['id']].join(df_preds, how='left').fillna(0)
submission['unit_sales'] = np.clip(np.expm1(submission['unit_sales']), 0, 1000)
submission.to_csv('lgb_sub.csv', float_format='%.4f', index=None)

print("[INFO] Finished! ( ^ _ ^ ) V")
print("[INFO] Done in %f seconds." % (time() - t0))

# ====================================================================================== #