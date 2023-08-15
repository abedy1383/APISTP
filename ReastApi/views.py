# Foreign library
import hashlib # create hash sha256 
from time import time # create time stamp 
from json import dumps # Convert the dictionary file to JSON
from fastapi import status # Using code status 
from random import randint # create random data intiger 
from threading import Thread  # run multi Thread 
from pydantic import BaseModel # Base data dict 
from typing import Union , Any  # Data type
from cryptography.fernet import Fernet # Encryption of user keys 
from rest_framework.views import APIView   # create api viewer 
from rest_framework.response import Response   # response data in clint 
from django.core.handlers.wsgi import WSGIRequest # Base requets send clint 
from .models import (
    User , 
    Api , 
    HashUserApi , 
    Sentiment as Model_Sentiment
) 
from .forms import (
    CreateUserForm , 
    CreateApiForm , 
    CreateSentiment , 
    PredictSentimentText , 
    FormUpdateHash , 
    FormLoginUser , 
    FormRicaweryDataUser , 
    FormChangeDataUser , 
    FormChangeEmailUser , 
    FormChangeUsernameUser
) 
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

# Nural_network = main.RunTester()
Nural_network = None

# Base Data Requests -> Form -> Base
class Base:
    '''  
    ## run code 
    >>> Base(**forms.Form(requests.POST).cleand_data) 
    
    ### output code 
    -> Base Data
    '''
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

    class BaseUpdateHash(BaseModel):
        hash_login : str 

    class BaseLoginUser(BaseModel):
        username : str 
        password : str 

# create hash data 
class CreateHashPassword:
    ''' 
    ### run code :
    >>> CreateHashPassword({"username " :  "abolfazl"}).run()

    ### output code :
    
       '''
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

# viewer create user 
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
        
    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in CreateUserForm.base_fields.items()] ,  
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

# viewer create Api 
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

    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in CreateApiForm.base_fields.items()] ,  
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

# viewer sentiment text prsion 
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

    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in CreateSentiment.base_fields.items()] ,  
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

# viewer show data Sentiment 
class Predict(APIView):
    def MultiDelet(self , _filter):
        def run():
            _filter.delete()
        return Thread(target=run).start()

    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs): 
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in PredictSentimentText.base_fields.items()] ,  
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

# viewer update hash login  
class UpdateHashLogin(APIView):
    def MultiUpdate(self , filter : list[HashUserApi] , username : str , Ip : str , UserAgent  : str ):
        Hash = CreateHashPassword({"username" : username , "Ip" : Ip , "UserAgent" : UserAgent}).run()
        def run():
            print(filter.update(Hash = Hash))

        Thread(target=run).start()
        return Hash 

    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in FormUpdateHash.base_fields.items()] ,  
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK) 
    
    def post(self, request : WSGIRequest , *args, **kwargs):
        data = FormUpdateHash(request.POST)

        if data.is_valid():
            claned = Base.BaseUpdateHash(**data.cleaned_data)

            if len(_filter := HashUserApi.objects.filter(Hash = claned.hash_login , Ip = request.get_host())) > 0:
                return Response(
                    Base.BaseResponseJson(
                        massage="ok update Hash" , 
                        data= {
                            "username" : _filter[0].user.username , 
                            "hash_login" : self.MultiUpdate(
                                filter= _filter, 
                                username= _filter[0].user.username, 
                                Ip= _filter[0].Ip, 
                                UserAgent= _filter[0].UserAgent,  
                            ) , 
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
                            "Block Ip or not Api"
                        ]
                    } , 
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                ).dict()
                , status=status.HTTP_200_OK) 
        return Response(
            Base.BaseResponseJson(
                massage="not update Hash" , 
                data= {
                    "errors" : data.errors
                } , 
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).dict()
            , status=status.HTTP_200_OK) 

# viewer login user  
class LoginUser(APIView):
    def multiThread(self , user : User , username : str , Ip : str , UserAgent : str  ):
        Hash = CreateHashPassword({"username" : username , "Ip" : Ip , "UserAgent" : UserAgent}).run()

        def run():
            HashUserApi(
                user = user , 
                Hash = Hash , 
                Ip = Ip,
                UserAgent = UserAgent , 
            ).save()
        Thread(target=run).start()
        return Hash
        
    @method_decorator(cache_page(7200))
    def get(self, request : WSGIRequest , *args, **kwargs):
        return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in FormLoginUser.base_fields.items()] ,  
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK) 
    
    def post(self, request : WSGIRequest , *args, **kwargs):
        data = FormLoginUser(request.POST)

        if data.is_valid():
            claned = Base.BaseLoginUser(**data.cleaned_data)

            if len(_filter := User.objects.filter(username = claned.username , password = claned.password )) > 0:
                return Response(
                    Base.BaseResponseJson(
                        massage="ok login user " , 
                        data= {
                            "username" : _filter[0].username , 
                            "hash_login" : self.multiThread(
                                user = _filter[0], 
                                username= _filter[0].username, 
                                Ip= request.get_host(), 
                                UserAgent=  request.headers.get('User-Agent') ,  
                            ) , 
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
                            "Block Ip or not Api"
                        ]
                    } , 
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                ).dict()
                , status=status.HTTP_200_OK) 
        return Response(
            Base.BaseResponseJson(
                massage="not update Hash" , 
                data= {
                    "errors" : data.errors
                } , 
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            ).dict()
            , status=status.HTTP_200_OK) 

# viwer register 
class RegisterUserAcount:
    class Base:
        class BaseRicaweryDataUser(BaseModel):
            username : str  = None
            password : str  = None
            email    : str  = None
        
            def permission(self):
                self.username = None if self.username is None or self.username == '' else self.username
                self.password = None if self.password is None or self.password == '' else self.password
                self.email = None if self.email is None or self.email == '' else self.email

        class BaseChangeDataUser(BaseModel):
            hash_login : str        
            password : str 

        class BaseChangeEmailUser(BaseModel):
            hash_login : str        
            email : str 

        class BaseChangeUsernameUser(BaseModel):
            hash_login : str        
            username : str 

    class Serialize:
        pass  

    class ChangePasswordUser(APIView):
        def MultiUpdate(self , filter : list[HashUserApi] , password : str ):
            def run(): 
                if (_Data := User.objects.filter(username = filter[0].user.username))[0].password != password:
                    _Data.update(password = password)

            Thread(target=run).start()

            return filter[0].user.username 

        @method_decorator(cache_page(7200))
        def get(self, request : WSGIRequest , *args, **kwargs):
            return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in FormChangeDataUser.base_fields.items()] ,  
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK) 

        def post(self, request : WSGIRequest , *args, **kwargs):
            data = FormChangeDataUser(request.POST)

            if data.is_valid():
                claned = RegisterUserAcount.Base.BaseChangeDataUser(**data.cleaned_data)
                
                if len(_filter := HashUserApi.objects.filter(Hash = claned.hash_login , Ip = request.get_host())) > 0:

                    return Response(
                        Base.BaseResponseJson(
                            massage="ok update password" , 
                            data= {
                                "username" : self.MultiUpdate(_filter , claned.password) , 
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
                                "Block Ip or not Api"
                            ]
                        } , 
                        status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                    ).dict()
                    , status=status.HTTP_200_OK) 
            
            return Response(
                Base.BaseResponseJson(
                    massage="not update Hash" , 
                    data= {
                        "errors" : data.errors
                    } , 
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                ).dict()
                , status=status.HTTP_200_OK) 

    class ChangeEmailUser(APIView):
        def MultiUpdate(self , filter : list[HashUserApi] , email : str ):
            def run(): 
                User.objects.filter(username = filter[0].user.username).update(email = email)

            Thread(target=run).start()

            return filter[0].user.username 

        @method_decorator(cache_page(7200))
        def get(self, request : WSGIRequest , *args, **kwargs):
            return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in FormChangeEmailUser.base_fields.items()] ,  
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK) 
        
        def post(self, request : WSGIRequest , *args, **kwargs):
            data = FormChangeEmailUser(request.POST)

            if data.is_valid():
                claned = RegisterUserAcount.Base.BaseChangeEmailUser(**data.cleaned_data)
                
                if len(_filter := HashUserApi.objects.filter(Hash = claned.hash_login , Ip = request.get_host())) > 0:

                    return Response(
                        Base.BaseResponseJson(
                            massage="ok update password" , 
                            data= {
                                "username" : self.MultiUpdate(_filter , claned.email) , 
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
                                "Block Ip or not Api"
                            ]
                        } , 
                        status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                    ).dict()
                    , status=status.HTTP_200_OK) 
            
            return Response(
                Base.BaseResponseJson(
                    massage="not update Hash" , 
                    data= {
                        "errors" : data.errors
                    } , 
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                ).dict()
                , status=status.HTTP_200_OK) 

    class ChangeUsernameUser(APIView):
        def MultiUpdate(self , filter : list[HashUserApi] , username : str ):
            def run(): 
                User.objects.filter(username = filter[0].user.username).update(username = username)

            Thread(target=run).start()

            return username

        @method_decorator(cache_page(7200))
        def get(self, request : WSGIRequest , *args, **kwargs):
            return Response(
            Base.BaseResponseJson(
                massage="This route is only for testing with the post method" , 
                data= {
                    "validateor" : [
                        {
                            _key : {
                                "max_length" : _item.max_length , 
                                "required" : _item.required , 
                                "type" : _item.__str__().split(".")[-1].split(" ")[0]
                            }
                        } for _key , _item in FormChangeUsernameUser.base_fields.items()] ,  
                } , 
                status_code=status.HTTP_204_NO_CONTENT
            ).dict()
            , status=status.HTTP_200_OK) 
 
        def post(self, request : WSGIRequest , *args, **kwargs):
            data = FormChangeUsernameUser(request.POST)

            if data.is_valid():
                claned = RegisterUserAcount.Base.BaseChangeUsernameUser(**data.cleaned_data)
                
                if len(_filter := HashUserApi.objects.filter(Hash = claned.hash_login , Ip = request.get_host())) > 0:

                    return Response(
                        Base.BaseResponseJson(
                            massage="ok update password" , 
                            data= {
                                "username" : self.MultiUpdate(_filter , claned.username) , 
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
                                "Block Ip or not Api"
                            ]
                        } , 
                        status_code=status.HTTP_405_METHOD_NOT_ALLOWED
                    ).dict()
                    , status=status.HTTP_200_OK) 
            
            return Response(
                Base.BaseResponseJson(
                    massage="not update Hash" , 
                    data= {
                        "errors" : data.errors
                    } , 
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE
                ).dict()
                , status=status.HTTP_200_OK) 

