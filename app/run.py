import json
import pandas as pd
import os


from flask import Flask
from flask import render_template, request, jsonify

import pickle

import numpy as np

import subprocess
import sys
import json
import requests

import yaml
import re

import gc
import pickle

import io
import codecs


from datetime import datetime, timedelta

from scipy import stats

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import f_regression


def round_significant(number,n_round):
    if number>0:
        round_number = np.round(number, np.int((n_round - 1 - (np.floor(np.log10(number))))) )
    else:
        round_number=0
    
    return round_number






app = Flask(__name__)


# load model
with open("model_storage.pkl",'rb') as file:
    df_model_storage = pickle.load(file,encoding='latin1')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    geo_map=df_model_storage[['province','city','kecamatan']]

    return render_template('master.html')


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # get all user input data
    province = request.args.get('province', '') 
    city = request.args.get('city', '')
    kecamatan = request.args.get('kecamatan', '')
    #get house type
    sqr_building = int(request.args.get('sqr_building', '0'))
    sqr_land = int(request.args.get('sqr_land', '0'))
    bedroom = int(request.args.get('bedroom', '0'))
    bathroom = int(request.args.get('bathroom', '0'))
    floor = int(request.args.get('floor', '0'))
    #get certificate
    shm = int(request.args.get('shm', '0'))
    hgb = int(request.args.get('hgb', '0'))
    lainnyappjbgirikadatdll = int(request.args.get('lainnya', '0'))
    #get facility
    ac= int(request.args.get('ac', '0'))
    swimmingpool = int(request.args.get('swimmingpool', '0'))
    carport= int(request.args.get('carport', '0'))
    garden= int(request.args.get('garden', '0'))
    garasi= int(request.args.get('garasi', '0'))
    telephone= int(request.args.get('telephone', '0'))
    pam=int(request.args.get('pam', '0'))
    waterheater=int(request.args.get('waterheater', '0'))
    refrigerator=int(request.args.get('refrigerator', '0'))
    stove=int(request.args.get('stove', '0'))
    microwave=int(request.args.get('microwave', '0'))
    oven= int(request.args.get('oven', '0'))
    fireextenguisher= int(request.args.get('fireextenguisher', '0'))
    gordyn=int(request.args.get('gordyn', '0'))
    
    sqr_land_log = np.log(sqr_land)
    sqr_building_log = np.log(sqr_building)

    user_features = [[floor,bedroom,bathroom,ac,swimmingpool,carport,garden,garasi,telephone,pam
                    ,waterheater,refrigerator,stove,microwave,oven,fireextenguisher,gordyn,hgb,lainnyappjbgirikadatdll
                    ,shm,sqr_land_log,sqr_building_log]]

    model_package = df_model_storage[(df_model_storage['province']==province)&(df_model_storage['city']==city)&(df_model_storage['kecamatan']==kecamatan)]
    get_model = model_package['lm_model'].values[0]

    price_prediction_log = get_model.predict(user_features)
    price_prediction = np.exp(price_prediction_log[0])
    ceil_preds=np.exp(price_prediction_log*1.01)[0]
    floor_preds=np.exp(price_prediction_log*0.99)[0]


    log_bawah = np.log(model_package['q25'].values[0])
    log_atas= np.log(model_package['q75'].values[0])
    IQR = log_atas - log_bawah
    limit_bawah = max(np.exp(log_bawah - 1.5*IQR), model_package['minimum'].values[0])
    limit_atas = min(np.exp(log_atas + 1.5*IQR), model_package['maximum'].values[0])

    print('Lokasi: '+model_package['kecamatan'].values[0]+', '+model_package['city'].values[0]+', '+model_package['province'].values[0])
    print('Rata-rata Harga Rumah: {:,}'.format(round_significant(model_package['mean'].values[0],2)))
    print('Kisaran Batas Harga Bawah: {:,}'.format(round_significant(limit_bawah,2)))
    print('Kisaran Batas Harga Atas: {:,}'.format(round_significant(limit_atas,2)))

    print('Harga Rumah Anda: {:,}'.format(round_significant(price_prediction,2)))
    print('Range Harga Anda:{:,} - {:,}'.format(round_significant(floor_preds,2),round_significant(ceil_preds,2)))

    print('Akurasi: {:.2f} %'.format(model_package['r2_score_test'].values[0]*100))


    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        price_prediction='{:,}'.format(round_significant(price_prediction,2)),
        floor_preds='{:,}'.format(round_significant(floor_preds,2)),
        ceil_preds='{:,}'.format(round_significant(ceil_preds,2)),

        accuracy='{:.2f} %'.format(model_package['r2_score_test'].values[0]*100),

        kecamatan=model_package['kecamatan'].values[0],
        city=model_package['city'].values[0],
        province=model_package['province'].values[0],

        mean='{:,}'.format(round_significant(model_package['mean'].values[0],2)),
        min_price='{:,}'.format(round_significant(limit_bawah,2)),
        q1='{:,}'.format(round_significant(model_package['q25'].values[0],2)),
        q2='{:,}'.format(round_significant(model_package['q50'].values[0],2)),
        q3='{:,}'.format(round_significant(model_package['q75'].values[0],2)),
        max_price='{:,}'.format(round_significant(limit_atas,2))

    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()