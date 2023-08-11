from django.db import models
from datetime import datetime 
from django.utils import timezone

class User(models.Model):
    id = models.AutoField(unique=True , primary_key=True)
    username = models.CharField(max_length=20 , unique=True) 
    password = models.CharField(max_length=20 ) 
    email = models.EmailField(unique=True)

    def __str__(self) -> str:
        return self.username

class HashUserApi(models.Model):
    id = models.AutoField(unique=True , primary_key=True)
    user = models.ForeignKey(User , on_delete=models.CASCADE)
    Hash = models.TextField()
    DataCreate = models.DateTimeField(auto_now_add=True)
    Ip = models.CharField(max_length=15)
    UserAgent = models.TextField()

    def __str__(self) -> str:
        return self.user.username

class Api(models.Model):
    id = models.AutoField(unique=True , primary_key=True)
    user_content = models.ForeignKey(User , on_delete=models.CASCADE)
    user_login = models.ForeignKey(HashUserApi , on_delete=models.CASCADE)
    api = models.TextField()
    use = models.IntegerField(default=0)

    def __str__(self) -> str:
        return self.user_content.username

class Sentiment(models.Model):
    id = models.AutoField(unique=True , primary_key=True)
    text = models.TextField()
    sentiment = models.CharField(max_length=20)
    code = models.CharField(max_length=40)
    api = models.ForeignKey(Api , on_delete=models.CASCADE)
    
    def __str__(self) -> str:
        return self.code
