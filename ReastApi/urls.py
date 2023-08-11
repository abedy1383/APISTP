from rest_framework import routers
from django.urls import path , include
from .views import UserCreate , ApiCreate , Sentiment , Predict , UpdateHashLogin , LoginUser , RegisterUserAcount , views

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('create/user/' , UserCreate.as_view()),
    path('create/api/' , ApiCreate.as_view()),
    path('create/sentiment/api/' , Sentiment.as_view()),
    path('show/sentiment/api/' , Predict.as_view()),
    path('update/hash/api/' , UpdateHashLogin.as_view()),
    path('login/user/api/' , LoginUser.as_view()),
    path('register/Change/password/api/' , RegisterUserAcount.ChangePasswordUser.as_view()),
    path('register/Change/email/api/' , RegisterUserAcount.ChangeEmailUser.as_view()),
    path('register/Change/username/api/' , RegisterUserAcount.ChangeUsernameUser.as_view()),

    path('home/' , views)
]


