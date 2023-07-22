from django.db import models

class User(models.Model):
    username = models.CharField(max_length=20 , unique=True) 
    password = models.CharField(max_length=20 ) 
    email = models.EmailField(unique=True)

    def __str__(self) -> str:
        return self.username

class HashUserApi(models.Model):
    user = models.ForeignKey(User , on_delete=models.CASCADE)
    Hash = models.TextField()
    DataCreate = models.DateTimeField(auto_now_add=True)
    Ip = models.CharField(max_length=15)
    UserAgent = models.TextField()

    def __str__(self) -> str:
        return self.user.username

class Api(models.Model):
    user_content = models.ForeignKey(User , on_delete=models.CASCADE)
    user_login = models.ForeignKey(HashUserApi , on_delete=models.CASCADE)
    api = models.TextField()
    use = models.IntegerField(default=0)

    def __str__(self) -> str:
        return self.user_content.username

class Sentiment(models.Model):
    text = models.TextField()
    code = models.IntegerField()
    api = models.ForeignKey(Api , on_delete=models.CASCADE)
    user = models.ForeignKey(User , on_delete=models.CASCADE)
    

    def __str__(self) -> str:
        return self.code




