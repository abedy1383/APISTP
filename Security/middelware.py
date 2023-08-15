from typing import Any
from .DataBase import database , Model , Session
from threading import Thread , Lock
from cryptography.fernet import Fernet
from functools import lru_cache
from django.core.handlers.wsgi import WSGIRequest
from rest_framework.response import Response

class ControllerHash:
    def __init__(self , 
        max_Thread : int = 500
    ) -> None:
        (
            self._LocShowkHash , 
            self._ListHash , 
            self.Max_Thread
        ) = (
            Lock() , 
            [] , 
            max_Thread 
        )

    def _registerHash(self):
        return Fernet.generate_key().decode()

    def _CreateHash(self):
        self._ListHash.append(self._registerHash())
 
    def __call__(self) -> Any:
        Thread(target=self._run).start()

    def _run(self):
        for _ in range(self.Max_Thread):
            self._CreateHash() 

    def _OperatorHash(self):
        Thread(target=self._CreateHash).start()
        return self._registerHash() 

    @property
    def Hash(self):
        self._LocShowkHash.acquire()
        try:
            return self._ListHash.pop(0) if len(self._ListHash) > 0 else self._OperatorHash() 
        finally:
            self._LocShowkHash.release() 

class ControllerDataBase:
    def __init__(self) -> None:
        database.Base.metadata.create_all(bind=database.engine)
        (
            self._db , 
            self._LockSessionDataBase
        ) = (
            database.SessionLocal , 
            Lock()
        )

    #model controller
    def _CreateSession(self) -> tuple :
        def run(db : Session):
            def _CloseContentDataBase():
                db.close()
                self._LockSessionDataBase.release()
            return _CloseContentDataBase

        self._LockSessionDataBase.acquire()
        return (db := self._db()) , run(db)
        
    def _CreateSesssionRead(self) -> Session:
        return self._db()

    #CSrf model controller 
    def _SaveCsrfToken(self , Ip : str , Headers : str , Hash : str ):
        _db , _closer = self._CreateSession()
        _db.add(Model.CsrfIpData(Hash = Hash , Headers = Headers , Ip = Ip ))
        _db.commit()
        _closer()
    
    def _CheckCsrfToken(self , Ip : str , Headers : str): 
        return True if self._CreateSesssionRead().query(
                Model.CsrfIpData
            ).filter(
                Model.CsrfIpData.Ip == Ip , 
                Model.CsrfIpData.Headers == Headers
            ).first() else False 

    def _UpdateCsrfToken(self , Ip : str , Headers : str , Hash : str ):
        _db , _closer = self._CreateSession()
        _db.query(
            Model.CsrfIpData
        ).filter(
            Model.CsrfIpData.Ip == Ip , 
            Model.CsrfIpData.Headers == Headers , 
        ).update({
            Model.CsrfIpData.Hash : Hash  
        })
        _db.commit()
        _closer()

    def _ShowCsrfToken(self , Ip :str , Headers : str ):
        return self._CreateSesssionRead().query(
                Model.CsrfIpData
            ).filter(
                Model.CsrfIpData.Ip == Ip , 
                Model.CsrfIpData.Headers == Headers
            )[0].Hash 

    # Block Ip Controller model 
    def _SaveIpBlock(self , Ip : str ):
        _db , _closer = self._CreateSession()
        _db.add(Model.BlockIp(Ip = Ip))
        _db.commit()
        _closer()
    
    def _CheckIpBlock(self , Ip : str ):
        return True if self._CreateSesssionRead().query(
                Model.BlockIp
            ).filter(
                Model.BlockIp.Ip == Ip , 
            ).first() else False 

    def _ShowAllIpBlock(self):
        _list = []
        if len(_data := self._CreateSesssionRead().query(Model.BlockIp).all()) > 0 :
            for _ in _data :
                _list.append(_.Ip)
        return _list

    # def _ControllerData(self):
    #     _db , closer = self._CreateSession()
    #     d = Model.CsrfIpData.delete().where(Model.CsrfIpData.TimeUpdated == 1)
    #     d.execute()

class ControllerDataIp:
    def __init__(self , 
        ip : str , 
        user_again : str , 
        controllerHash : ControllerHash , 
        controllerDataBase : ControllerDataBase 
    ) -> None:
        (
            self._ip , 
            self._user_again , 
            self._ControllerHash , 
            self._ControllerDataBase
        ) = (
            ip , 
            user_again , 
            controllerHash , 
            controllerDataBase
        )

        #run function 
        self._CallController()

    def _CallController(self) -> None :
        #1
        if not self._ControllerDataBase._CheckCsrfToken(
            self._ip , 
            self._user_again
        ):
            self._hash , self._CreateNew = self._ControllerHash.Hash , True
            self._ControllerDataBase._SaveCsrfToken(
                self._ip , 
                self._user_again , 
                self._hash
            )

        else:
            self._hash , self._CreateNew = self._ControllerDataBase._ShowCsrfToken( self._ip , self._user_again ) , False 

        return None 
      
    @property
    def Hash(self):
        if self._CreateNew :
            self._CreateNew = False 
            return self._hash 
        else:
            self._ControllerHash._ListHash.append(self._hash)
            self._hash = self._ControllerHash.Hash 
            Thread(target=self._ControllerDataBase._UpdateCsrfToken , args=(self._ip , self._user_again , self._hash ,)).start()
            return self._hash 
      
    @property
    def Show_Dict(self):
        return {
            "Ip" : self._ip , 
            "Hash" : self._hash , 
            "User_Again" : self._user_again 
        }

class ControllerBlockIp:
    def __init__(self , 
        controllerDataBase : ControllerDataBase
    ) -> None:
        (
            self._ListBlockIp , 
            self._ControllerDataBase
        ) = (
            controllerDataBase._ShowAllIpBlock() , 
            controllerDataBase
        )

    def _Save(self , ip ):
        if ip not in self._ListBlockIp:
            self._ListBlockIp.append(ip)
            self._ControllerDataBase._SaveIpBlock(ip)

    @property
    def Block(self):
        return self._ListBlockIp
    
    @Block.setter 
    def Block(self , ip : str  ):
        return Thread(target=self._Save , args=(ip,)).start()

class controllerMultiCreate:
    def __init__(self) -> None:
        self._listCreatorData = []

    def _Deleter(self , value):
        def run():
            self._listCreatorData.pop(self._listCreatorData.index(value))
        return run 

    def setter(self , value):
        self._listCreatorData.append(value)
        return self._Deleter(value)

class MiddlewareSecurityClint:
    def __init__(self , 
        get_response 
    ) -> None:
        (
            self.get_response , 
            self._ControllerHash , 
            self._controllerDataBase , 
            self._ControllerMultiCreateData , 
            self._DataBase , 
            self._TriningIpData
        ) = (
            get_response , 
            ControllerHash() , 
            ControllerDataBase() , 
            controllerMultiCreate() , 
            {} , 
            []
        )

        self._ControllerBlockerIp = ControllerBlockIp(self._controllerDataBase)

        #run function 
        self._ControllerHash()
    
    @lru_cache
    def _ResponseBlock(self):
        response = Response()
        response.content = "404_Not_Found"
        return response

    def _CreateControllerIp(self , ip : str , headers : str , response : Response ):
        if ip not in self._ControllerMultiCreateData._listCreatorData:
            _deleter = self._ControllerMultiCreateData.setter(ip)
            self._DataBase.update({
                ip : {
                    headers : (_data := ControllerDataIp(ip , headers , self._ControllerHash , self._controllerDataBase))
                }
            })
            _deleter()
            return response.set_cookie(key="HASH" , value=_data._hash) 
        
    def _CheckIp(self , requests : WSGIRequest , response : Response ):
        (
            _ip , 
            _headers
        ) = (
            requests.get_host().split(":")[0] , 
            requests.headers.get('User-Agent')
        )

        if _ip in self._DataBase.keys() and _headers in self._DataBase[_ip].keys():
            if self._DataBase[_ip][_headers]._hash == requests.COOKIES.get("HASH"):
                return response.set_cookie(key="HASH" , value=self._DataBase[_ip][_headers].Hash)
            else:
                self._ControllerBlockerIp.Block = _ip
                return False 
            
        return self._CreateControllerIp(_ip , _headers , response)

    def __call__(self, request : WSGIRequest) -> Response:
        if request.get_host().split(":")[0] not in self._ControllerBlockerIp._ListBlockIp:
            self._CheckIp(request , (response := self.get_response(request)))
            return response  
        return self._ResponseBlock()

# ست کردن زمان هش 
