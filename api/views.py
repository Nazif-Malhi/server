from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from django.http import JsonResponse
import json
from django.conf import settings
import os

base_dir = settings.MEDIA_ROOT
my_file = os.path.join(base_dir, str('crops_recomendation_model1.pickle'))

pickle_in = open(
    my_file, 'rb')
crops_recomendation_model1 = pickle.load(pickle_in)
crops_with_soil_df1 = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Crops/Crop_with_soil%20(i).csv")
le1 = preprocessing.LabelEncoder()


crops_with_soil_df1['soil'] = le1.fit_transform(crops_with_soil_df1['soil'])


def crop_recomendation_prediction(request):
    data = json.loads(request.body)
    print(data['soil_type'])
    convertedLabel = le1.transform([data['soil_type']])
    prepare_data = np.array(
        [[convertedLabel, float(data['temp']), float(data['humi']), float(data['ph']), float(data['rain'])]])
    prediction_crop = crops_recomendation_model1.predict(prepare_data)
    return JsonResponse({"prediction": prediction_crop[0]})
