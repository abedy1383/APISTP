from rest_framework import routers
from django.urls import path , include
from .views import UserCreate , ApiCreate , Sentiment , Predict

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('create/user/' , UserCreate.as_view()),
    path('create/api/' , ApiCreate.as_view()),
    path('create/sentiment/api/' , Sentiment.as_view()),
    path('show/sentiment/api/' , Predict.as_view()),
]
