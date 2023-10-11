
from json import loads
from pathlib import Path
from torch.backends import cudnn 
from torch import cuda , manual_seed 
import logging , os , torch 
import logging 

from .Datasets.analized import ClanedText 
from .Model.Model.BertModelSTP import SentimentModel 

manual_seed(9999)
cuda.manual_seed(9999)
cudnn.deterministic = True

class Setting:
    _addresEn = "HooshvareLab/bert-fa-base-uncased"
    _addres_folder_import = Path(__file__).resolve().parent / "Model/Saved"
    _addres_commants = "negative"
    _addres_swear = "negativeplas"   #"negativeplas"

class Tester:
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

    class Controller:
        class Model:
            def __call__(self, addres_folder : str , device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> SentimentModel:
                self.model = SentimentModel("HooshvareLab/bert-fa-base-uncased" , ['negative', 'positive']).to(device)
                if os.access(addres_folder , 0):

                    if os.access(f"{addres_folder}/model/model.pt" , 0):
                        self.model._LoadModel(f"{addres_folder}/model/model.pt")

                    elif os.access(f"{addres_folder}/model/model.json" , 0):
                        _json = loads(open(f"{addres_folder}/model/model.json" , 'r').read())
                        self.model._Dawnload(_json["file_id"] , _json['dest_path'])

                    else:
                        logging.warning("Not Loading Model : [ERROR]")
                return self.model 

    def __init__(self ,
            Comments_folder : str = Setting._addres_commants,
            Swear_folder : str = Setting._addres_swear, 
            addres_import : str = Setting._addres_folder_import 
        ) -> None:

        # تشخسص تظر کاربران
        self.Comments = Tester.Controller.Model()(
            addres_folder=f"{addres_import}/{Comments_folder}"
        )

        # تشخیص فش
        self.Swear = Tester.Controller.Model()(
            addres_folder=f"{addres_import}/{Swear_folder}"
        )

        self._clanedtext = ClanedText() 
    
    def predict(self , text):
        if not self.Comments.predict(text ).item():
            if not self.Swear.predict(text).item():
                return Tester.Base.UnNegative()
            return Tester.Base.Negative()
        return Tester.Base.Positive()


        
