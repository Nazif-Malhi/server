from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from django.http import JsonResponse
import json
from django.conf import settings
import os
from sklearn.preprocessing import OneHotEncoder

base_dir = settings.MEDIA_ROOT
file_path_crop1 = os.path.join(
    base_dir, str('crops_recomendation_model1.pickle'))
file_path_crop2 = os.path.join(
    base_dir, str('crops_recomendation_model2.pickle'))
file_path_fertilizer = os.path.join(
    base_dir, str('fertilizer_recomendation_model.pickle'))


crop1_pickle_in = open(
    file_path_crop1, 'rb')
crop2_pickle_in = open(
    file_path_crop2, 'rb')
fertilizer_pickle_in = open(
    file_path_fertilizer, 'rb')


crops_recomendation_model1 = pickle.load(crop1_pickle_in)
crops_recomendation_model2 = pickle.load(crop2_pickle_in)
fertilizer_recomendation_model = pickle.load(fertilizer_pickle_in)


crops_with_soil_df1 = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Crops/Crop_with_soil%20(i).csv")
crops_with_soil_df2 = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Crops/Crop_with_soil%20(ii).csv")
fertilizer_df = pd.read_csv(
    "https://raw.githubusercontent.com/Nazif-Malhi/Farmstead_Models/main/ML%20Models%20Farmstead/Dataset/Fertilizer/Fertilizer%20Prediction.csv")


le1 = preprocessing.LabelEncoder()
crops_with_soil_df1['soil'] = le1.fit_transform(crops_with_soil_df1['soil'])

le2 = preprocessing.LabelEncoder()
crops_with_soil_df2['soil'] = le2.fit_transform(crops_with_soil_df2['soil'])

le_soil = preprocessing.LabelEncoder()
le_crop = preprocessing.LabelEncoder()
fertilizer_df['Soil Type'] = le_soil.fit_transform(fertilizer_df['Soil Type'])
fertilizer_df['Crop Type'] = le_crop.fit_transform(fertilizer_df['Crop Type'])


def crop_simple_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabel = le1.transform([data['soil_type']])
    prepare_data = np.array(
        [[convertedLabel, float(data['temp']), float(data['humi']), float(data['ph']), float(data['rain'])]])
    prediction_crop1 = crops_recomendation_model1.predict(prepare_data)
    return JsonResponse({"prediction_simple": prediction_crop1[0]})


def crop_advance_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabel = le2.transform([data['soil_type']])
    prepare_data = np.array(
        [[float(data['nitrogen']), float(data['phosphorus']), float(data['potassium']), convertedLabel, float(data['temp']), float(data['humi']), float(data['ph']), float(data['rain'])]])
    prediction_crop2 = crops_recomendation_model2.predict(prepare_data)
    return JsonResponse({"prediction_advance": prediction_crop2[0]})


def fertilizer_recomendation_prediction(request):
    data = json.loads(request.body)
    convertedLabelSoil = le_soil.transform([data['soil']])
    convertedLabelCrop = le_crop.transform([data['crop']])
    data = np.array([[float(data['temp']), float(data['humi']), float(data['moisture']), convertedLabelSoil,
                    convertedLabelCrop, float(data['nitrogen']), float(data['phosphorus']), float(data['potassium'])]])
    prediction_fertilizer = fertilizer_recomendation_model.predict(data)
    return JsonResponse({"prediction_fertilizer": prediction_fertilizer[0]})
