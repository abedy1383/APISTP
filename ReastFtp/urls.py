from django.urls import path 
from .views import ( 
    ShowHome , 
    ShowSignUp , 
    ShowSignIn 
)

urlpatterns = [
    path('' , ShowHome.as_view()) , 
    path('signin/' , ShowSignIn.as_view()) , 
    path('signup/' , ShowSignUp.as_view()) , 

]
