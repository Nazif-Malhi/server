from django.urls import path
from api import views
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path('crop_recomendation1/', csrf_exempt(views.crop_recomendation_prediction))
]
