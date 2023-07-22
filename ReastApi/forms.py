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
    hash_api = forms.CharField(max_length=1000)
    hash_login = forms.CharField(max_length=1000)
 
