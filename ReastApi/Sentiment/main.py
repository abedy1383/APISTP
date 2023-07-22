import logging , os
import torch , pickle
from typing import Any 
import numpy as np
from pydantic import BaseModel 
from functools import lru_cache
from torch.backends import cudnn 
from torch import cuda , manual_seed 
from .Datasets.analized import DataLoder 
import pandas as pd
from .Datasets.analized import ClanedText 
from .Model.Model.BiLstm import SentimentModel 
from json import loads
from pathlib import Path
import requests
from bs4 import BeautifulSoup 

class Setting:
    _addresEn = "HooshvareLab/bert-fa-base-uncased"
    _driver = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _addres_folder_import = Path(__file__).resolve().parent / "Model/Saved"
    _addres_commants = "negative"
    _addres_swear = "negativeplas"

class Download:
    def run(id, destination):
        session = requests.Session()

        response = requests.post(BeautifulSoup(session.get("https://docs.google.com/uc?export=download", params = { 'id' : id }, stream = True).content, "html.parser").find(id="download-form").get("action"), stream=True)

        with open(destination, "wb") as handle:
          handle.write(response.content)

class RunTester:
    class Base:
        class Positive:
            def __str__(self) -> str:
                return "displayable (positive)"
            
        class UnNegative:
            def __str__(self) -> str:
                return "undisplayable (negative)"
        
        class Negative:
            def __str__(self) -> str:
                return "displayable (negative)"

    class Model:
        class BaseSetting(BaseModel):
            accury : Any 
            epoch : Any 
            step : Any 
            loss : Any 
            labels : Any 

        class BaseTokenizer(BaseModel):
            text : Any= None
            label : Any = None
            attention_mask : Any = None
            token_type_ids : Any = None

    class Controller:
        class ModelController:
            def __init__(self , 
                name_folder : str ,
                addres_import : str = Setting._addres_folder_import , 
                encoding : str = Setting._addresEn , 
                _driver : str = Setting._driver
                ) -> None:

                (
                    self.dictory , 
                    self.addres_import , 
                    self._encoding , 
                    self._driver 
                ) = (
                    name_folder , 
                    addres_import , 
                    encoding , 
                    _driver
                )

                # run function 
                self.run()

            def _setlogger(self):
                logging.basicConfig(filename=f"{self.addres_import}/{self.dictory}/log/myapp.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s' , force=True)

            def _checker_addres(self , addres , name ):
                if name in os.listdir(addres):
                    return True 
                return False 

            def _ShowSetting(self):
                if self._checker_addres(f"{self.addres_import}/{self.dictory}" , "Setting"):
                    if self._checker_addres(f"{self.addres_import}/{self.dictory}/Setting" , "Setting.pickle"):

                        logging.debug(f"show setting addres : {self.addres_import}/{self.dictory}/Setting")
                        with open(f"{self.addres_import}/{self.dictory}/Setting/Setting.pickle", 'rb') as handle:
                            return RunTester.Model.BaseSetting(**pickle.load(handle))

                return False

            def _checklossModel(self):
                if self._checker_addres(f"{self.addres_import}/{self.dictory}/model" , "model.pt"):
                    return self.model.load_state_dict(torch.load(f"{self.addres_import}/{self.dictory}/model/model.pt" , map_location=self._driver))
                else:
                    Download.run(
                        id=(_json := loads(open(f"{self.addres_import}/{self.dictory}/model/model.json", 'r').read()))["file_id"],
                        destination= f"{self.addres_import}/{self.dictory}/model/{_json['dest_path']}" 
                    )
                    return self._checklossModel()

            def _createmodel(self):
                self.model : SentimentModel = SentimentModel(self._encoding , self._ShowSetting().labels) 

                # automatic load model
                self._checklossModel()

            def _optim(self):
                self.model.to(self._driver)

            def run(self):
                self._setlogger()
                self._createmodel()
                self._optim()
        
    def __init__(self , 
            Comments_folder : str = Setting._addres_commants,
            Swear_folder : str = Setting._addres_swear, 
            addres_import : str = Setting._addres_folder_import , 
            encoding : str = Setting._addresEn , 
            _driver : str = Setting._driver
        ) -> None:

        # تشخسص تظر کاربران
        self.Comments = RunTester.Controller.ModelController(
                name_folder=Comments_folder , 
                addres_import=addres_import , 
                encoding=encoding , 
                _driver = _driver
            )

        # تشخیص فش
        self.Swear = RunTester.Controller.ModelController(
                name_folder=Swear_folder , 
                addres_import=addres_import , 
                encoding=encoding , 
                _driver = _driver
            )

        (
            self._driver , 
            self._clanedtext , 
            self._loader
        ) = (
            _driver , 
            ClanedText() , 
            DataLoder()
        )

    def _modelCommands(self , comments , controller : Any):

        predictions =  []

        controller.model.eval()

        with torch.no_grad():

            for data in self._loader.run( comments = comments ):

                BaseData = RunTester.Model.BaseTokenizer(**data)

                predictions.extend(
                    torch.max(
                        controller.model(
                            input_ids=BaseData.text.to(self._driver),
                            attention_mask=BaseData.attention_mask.to(self._driver),
                            token_type_ids=BaseData.token_type_ids.to(self._driver)
                        )
                    , dim=1)[1]
                )

        return torch.stack(predictions).cpu().detach().numpy()
    
    def _Clustering(self , text):
        # اضافه کردن خوشه بندی :)
        if not self._modelCommands(( _textnumpy := np.array([text]) ) , self.Comments)[0]:
            if not self._modelCommands( _textnumpy , self.Swear)[0]:
                return RunTester.Base.UnNegative()
            return RunTester.Base.Negative()
        return RunTester.Base.Positive()
    
    def predict(self , comments : list[str]):

        return pd.DataFrame(data={
            "Texts": (_claned := [self._clanedtext.predict(_) for _ in comments]),
            "Comments" :  [str(self._Clustering(_)) for _ in _claned]
        })
    
    def predict_one(self , text):
        return str(self._Clustering(text))
