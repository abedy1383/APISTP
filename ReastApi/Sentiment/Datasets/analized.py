import pandas as pd
from typing import Union
import re , hazm 
from numpy import ndarray
from cleantext import clean
from transformers import BertTokenizer 
from torch.utils.data import  DataLoader

class Setting:
    addres_CSV = "./Datasets/Csv/DataFrame.csv"
    addres_json = "./Datasets/json/data_negative.json"
    encode = 'HooshvareLab/bert-fa-base-uncased'
    random_seed = 9999
    batch_size = 64 

class ClanedText:
    def __init__(self) -> None:
        
        # run function 
        self._cash()

    def _cash(self):
        self._complet = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
                u"\u2069"
                u"\u2066"
                u"\u200c"
                u"\u2068"
                u"\u2067"
                "]+", flags=re.UNICODE)
        self._nomalize = hazm.Normalizer()
        self._clanedtext = re.compile('<.*?>')
        self._clanedzero = re.compile('[!@#$%^&*()_+=./<>{}-]')

    def predict(self , value : str ):
        return re.sub(r'\s*[A-Za-z]+\b', '' , 
                        re.sub("\s+", " ", 
                            re.sub("#", "", 
                                self._complet.sub(r'', 
                                    self._nomalize.normalize(
                                        re.sub( self._clanedtext, '', clean(
                                            self._clanedzero.sub("" , value).strip(),
                                            fix_unicode=True,
                                            to_ascii=False,
                                            lower=True,
                                            no_line_breaks=True,
                                            no_urls=True,
                                            no_emails=True,
                                            no_phone_numbers=True,
                                            no_numbers=False,
                                            no_digits=False,
                                            no_currency_symbols=True,
                                            no_punct=False,
                                            replace_with_url="",
                                            replace_with_email="",
                                            replace_with_phone_number="",
                                            replace_with_number="",
                                            replace_with_digit="0",
                                            replace_with_currency_symbol="",
                                            )
                                        )
                                    )                        
                                )                      
                            )      
                        )
                    ).rstrip()
        
class DataLoder: 
    def __init__(self , encoding : str = Setting.encode) -> None:
        self._tokenizer = BertTokenizer.from_pretrained(encoding)

    def _loader(self , 
            comment : list , 
            label : list , 
            batch_size : int
        ):
        
        return DataLoader(
            Datasets(
                comments = comment ,
                targets = label ,
                tokenizer = self._tokenizer
            ),
            batch_size=batch_size 
            )

    def run(self , comments : Union[pd.Series , ndarray ]  , labels : Union[pd.Series , None] = None , batch_size : int = 64 ):
        
        return self._loader(
            comment= comments.to_numpy() if not isinstance(comments , ndarray) else comments,
            label= labels if labels is None else labels.to_numpy() , 
            batch_size= batch_size,
        )
