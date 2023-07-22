from django.contrib import admin
from django.urls import path , include

from rest_framework import routers
from .views import UserCreate , ApiCreate , Sentiment

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('create/user/' , UserCreate.as_view()),
    path('create/api/' , ApiCreate.as_view()),
    path('sentiment/api/' , Sentiment.as_view()),

]
