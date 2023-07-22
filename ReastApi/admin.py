from django.contrib import admin
from .models import User , Api , HashUserApi , Sentiment

admin.site.register(User)
admin.site.register(Api)
admin.site.register(HashUserApi)
admin.site.register(Sentiment)