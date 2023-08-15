from rest_framework import routers
from django.urls import path , include
from .views import UserCreate , ApiCreate , Sentiment , Predict , UpdateHashLogin , LoginUser , RegisterUserAcount 

router = routers.DefaultRouter()

app_name = "UrlRestApi"

urlpatterns = [
    path('', include(router.urls)),
    path('create/user/' , UserCreate().as_view() , name="CreateUser"),
    path('create/api/' , ApiCreate.as_view() , name="CreateApi"),
    path('create/sentiment/' , Sentiment.as_view() , name="CreateSentiment"),
    path('show/sentiment/' , Predict.as_view() , name="ShowSentiment"),
    path('update/hash/' , UpdateHashLogin.as_view(), name="UpdateHashLogin"),
    path('login/user/' , LoginUser.as_view() , name="LoginUser"),
    path('register/Change/password/' , RegisterUserAcount.ChangePasswordUser.as_view() , name="RegisterUserChangePassword"),
    path('register/Change/email/' , RegisterUserAcount.ChangeEmailUser.as_view() , name="RegisterUserChangeEmail"),
    path('register/Change/username/' , RegisterUserAcount.ChangeUsernameUser.as_view() , name="RegisterUserChangeUsername"),
]


