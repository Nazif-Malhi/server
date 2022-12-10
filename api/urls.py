from django.urls import path
from api import views
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path('crop_recomendation_simple/',
         csrf_exempt(views.crop_simple_recomendation_prediction)),
    path('crop_recomendation_advance/',
         csrf_exempt(views.crop_advance_recomendation_prediction)),
    path('fertilizer_recomendation/',
         csrf_exempt(views.fertilizer_recomendation_prediction)),
]
