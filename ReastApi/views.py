import hashlib
from time import time 
from json import dumps
from fastapi import status
from random import randint
from .Sentiment import main 
from threading import Thread 
from pydantic import BaseModel
from typing import Union , Any 
from cryptography.fernet import Fernet
from rest_framework.views import APIView  
from rest_framework.response import Response  
from django.core.handlers.wsgi import WSGIRequest
from .models import User , Api , HashUserApi , Sentiment as Model_Sentiment 
from .forms import CreateUserForm , CreateApiForm , CreateSentiment , PredictSentimentText

Nural_network = main.RunTester()

class Base:
    class BaseResponseJson(BaseModel):
        massage : Union[str , None , list] = None 
        data : Union[dict[str , Any] , None] = None 
        status_code : int = 200 

    class BaseUserDataCreate(BaseModel):
        password : str 
        email : str 
        username : str 

    class BaseApiDataCreate(BaseModel):
        hash_login : str 

    class BaseApiSentiment(BaseModel):
        text : str 
        api : str 
    
    class BaseApiPedict(BaseModel):
        code : str 
        api : str 

class CreateHashPassword:
    def __init__(self, data: dict) -> None:
        (
            self._data , 
            self._key , 
            self._encrypted_password 
        ) = (
            dumps(data).encode() , 
            Fernet.generate_key() , 
            '' 
        )

    def _join(self , token , index):
        self._encrypted_password += chr(ord(token)+1)
        self._encrypted_password += chr(ord(self._key.decode()[index] if index <= len(list(self._key)) - 1 else str(randint(0, 9)))+1) 

    def run(self):
        [self._join(_token , _index) for _index , _token in enumerate(Fernet(self._key).encrypt(self._data).decode())] 

        return self._encrypted_password

class UserCreate(APIView):
    def multiThread(self , username : str , Ip : str , UserAgent : str  ):
        Hash = CreateHashPassword({"username" : username , "Ip" : Ip , "UserAgent" : UserAgent}).run()
        def run():
            HashUserApi(
                user = User.objects.get(username = username) , 
                Hash = Hash , 
                Ip = Ip,
                UserAgent = UserAgent , 
            ).save()
        Thread(target=run).start()
        return Hash
        
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "Inputs" : list(CreateUserForm.base_fields) , 
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK)  

    def post(self, request : WSGIRequest , *args, **kwargs):
            print(request.headers.get('User-Agent'))
            # loacal forms
            forms = CreateUserForm(request.POST)
            # valideision 
            if forms.is_valid():
                # save forms data
                forms.save()
                # create Base Data send User forms 
                claned = Base.BaseUserDataCreate(**forms.cleaned_data)
                # ok data
                return Response(
                    Base.BaseResponseJson(
                        massage="ok create account" , 
                        data= {
                            "username" : claned.username , 
                            "email" : claned.email , 
                            "hash_login" : self.multiThread(
                                username = claned.username , 
                                Ip= request.get_host() , 
                                UserAgent= request.headers.get('User-Agent')
                            ) , 
                            "timestamp" : time() , 
                        } , 
                        status_code=status.HTTP_201_CREATED
                    ).dict()
                    , status=status.HTTP_200_OK)  
            # not create 
            return Response(
                Base.BaseResponseJson(
                    massage="not create account user" , 
                    data= {
                        "errors" : forms.errors
                    } , 
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                ).dict()
                , status=status.HTTP_200_OK)  

class ApiCreate(APIView):
    def MultiThread(self , user_content , user_login ):
        api = CreateHashPassword({"time":time() , "Ip" : user_login.Ip }).run() 
        def run():
            Api(
                user_content = user_content, 
                user_login = user_login , 
                api = api
            ).save()

        Thread(target=run).start()
        return api

    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "data" : list(CreateApiForm.base_fields)
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK)  

    def post(self, request : WSGIRequest , *args, **kwargs):
        claned = CreateApiForm(request.POST)

        if claned.is_valid():
            data = HashUserApi.objects.filter(Hash = Base.BaseApiDataCreate(**claned.cleaned_data).hash_login)

            if len(data) > 0 :
                if data[0].Ip == request.get_host() and data[0].UserAgent == request.headers.get('User-Agent'):
                    return Response(
                        Base.BaseResponseJson(
                            massage="ok create api" , 
                            data= {
                                "username": data[0].user.username , 
                                "hash_api" : self.MultiThread( user_content= data[0].user , user_login= data[0]) ,
                                "timestamp" : time() , 
                            } , 
                            status_code=status.HTTP_201_CREATED
                        ).dict()
                        , status=status.HTTP_200_OK) 

                return Response(
                    Base.BaseResponseJson(
                        massage="You do not have access rights" , 
                        data= {
                            "errors" : [
                                "Block Ip"
                            ]
                        } , 
                        status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                    ).dict()
                    , status=status.HTTP_200_OK) 

            return Response(
                Base.BaseResponseJson(
                    massage="No such hash exists" , 
                    data= {
                        "errors" : [
                            "Not Found Hash Login"
                        ]
                    } , 
                    status_code=status.HTTP_203_NON_AUTHORITATIVE_INFORMATION
                ).dict()
                , status=status.HTTP_200_OK)  
        
        return Response(
            Base.BaseResponseJson(
                massage="not create account user" , 
                data= {
                    "errors" : claned.errors
                } , 
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).dict()
            , status=status.HTTP_200_OK)  
        
class Sentiment(APIView):
    def MultiSaveUse(self , _filter):
        def run():
            _filter.update(use = _filter[0].use + 1)
        return Thread(target=run).start()
    
    def MultiCreated(self , text , api ):
        code = hashlib.sha256(f"{time()}".encode('UTF-8')).hexdigest()
        def run():
            Model_Sentiment(
                text = text , 
                sentiment = Nural_network.predict_one(text), 
                code = code ,
                api = api, 
            ).save()

        Thread(target=run).start()
        return code 

    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "data" : list(CreateSentiment.base_fields)
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK)  

    def post(self, request : WSGIRequest , *args, **kwargs):
        claned = CreateSentiment(request.POST)

        if claned.is_valid():
            analized = Base.BaseApiSentiment(**claned.cleaned_data)
            if len(tokenapi:= Api.objects.filter(api = analized.api)) > 0 :
                self.MultiSaveUse(tokenapi)
                
                return Response(
                    Base.BaseResponseJson(
                        massage="ok sentiment text persion" , 
                        data= {
                            "username": tokenapi[0].user_content.username , 
                            "code" : self.MultiCreated(text=analized.text , api=tokenapi[0]), 
                            "timestamp" : time() , 
                        } , 
                        status_code=status.HTTP_200_OK
                    ).dict()
                    , status=status.HTTP_200_OK)    
            return Response(
                Base.BaseResponseJson(
                    massage="You do not have access rights" , 
                    data= {
                        "errors" : [
                            "Block Ip or not Api"
                        ]
                    } , 
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                ).dict()
                , status=status.HTTP_200_OK) 
        return Response(
            Base.BaseResponseJson(
                massage="not create account user" , 
                data= {
                    "errors" : claned.errors
                } , 
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).dict()
            , status=status.HTTP_200_OK) 

class Predict(APIView):
    def MultiDelet(self , _filter):
        def run():
            _filter.delete()
        return Thread(target=run).start()

    def get(self, request : WSGIRequest , *args, **kwargs): 
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "data" : list(PredictSentimentText.base_fields)
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK)  

    def post(self, request : WSGIRequest , *args, **kwargs):
        claned = PredictSentimentText(request.POST) 

        if claned.is_valid():
            data = Base.BaseApiPedict(**claned.cleaned_data)
            if len(_api := Api.objects.filter(api = data.api)) > 0 :
                if len(_data := Model_Sentiment.objects.filter(code = data.code , api = _api[0])) > 0 :
                    try:
                        return Response(
                            Base.BaseResponseJson(
                                massage="ok sentiment text persion" , 
                                data= {
                                    "username": _data[0].api.user_content.username , 
                                    "Sentiment" : _data[0].sentiment, 
                                    "timestamp" : time() , 
                                } , 
                                status_code=status.HTTP_200_OK
                            ).dict()
                            , status=status.HTTP_200_OK) 
                    finally:
                        self.MultiDelet(_data)
                        
                return Response(
                    Base.BaseResponseJson(
                        massage="not code" , 
                        data= {
                            "errors" : [
                                "not find data"
                            ]
                        } , 
                        status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                    ).dict()
                    , status=status.HTTP_200_OK) 
            return Response(
                Base.BaseResponseJson(
                    massage="You do not have access rights" , 
                    data= {
                        "errors" : [
                            "Block Ip or not Api"
                        ]
                    } , 
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                ).dict()
                , status=status.HTTP_200_OK) 
        return Response(
            Base.BaseResponseJson(
                massage="not create account user" , 
                data= {
                    "errors" : claned.errors
                } , 
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).dict()
            , status=status.HTTP_200_OK) 
