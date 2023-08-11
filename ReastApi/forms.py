from django import forms 
from .models import User , Sentiment

class CreateUserForm(forms.ModelForm):
    class Meta:
        model = User 
        fields = "__all__"  
 
class CreateApiForm(forms.Form):
    hash_login = forms.CharField(max_length=10000)

class CreateSentiment(forms.Form):
    text = forms.CharField(max_length=1000)
    api = forms.CharField(max_length=1000)
 
class PredictSentimentText(forms.Form):
    api = forms.CharField(max_length=1000)
    code = forms.CharField() 

class FormUpdateHash(forms.Form):
    hash_login = forms.CharField()

class FormLoginUser(forms.Form):
    username = forms.CharField(max_length=20)
    password = forms.CharField(max_length=40)
    
class FormRicaweryDataUser(forms.Form):
    username = forms.CharField(max_length=20 , required=False )
    password = forms.CharField(max_length=20 , required=False )
    email = forms.EmailField(max_length=20 , required=False )

class FormChangeDataUser(forms.Form):
    hash_login = forms.CharField()
    password = forms.CharField()

class FormChangeEmailUser(forms.ModelForm):
    hash_login = forms.CharField() 

    class Meta:
        model = User 
        fields = ["email"]

class FormChangeUsernameUser(forms.ModelForm):
    hash_login = forms.CharField() 

    class Meta:
        model = User 
        fields = ["username"]

    
