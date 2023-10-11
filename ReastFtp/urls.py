from django.urls import path 
from .views import ( 
    ShowHome , 
    ShowSignUp , 
    ShowSignIn 
)

app_name = "UrlReastFtp"

urlpatterns = [
    path('' , ShowHome.as_view() , name="Home") , 
    path('signin/' , ShowSignIn.as_view() , name="SignIn") , 
    path('signup/' , ShowSignUp.as_view() , name="SignUp") ,  
]
