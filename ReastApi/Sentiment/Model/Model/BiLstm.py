from torch import nn 
from transformers import BertConfig
from ReastApi.Sentiment.Model.Model.BiLinear import BertModel 

class Setting:
    encoding : str = 'HooshvareLab/bert-fa-base-uncased'

class Tokenizer():
    def __init__(self , 
                setting : dict , 
                path : str = Setting.encoding ,
            ) -> None:
        (
            self._path ,
            self._setting 
        ) = (
            path , 
            setting
        )

    @property 
    def Config(self):
        return BertConfig.from_pretrained(self._path, **self._setting)

class SentimentModel(nn.Module):
    def __init__(self, 
            encodeing : str = Setting.encoding , 
            setting : dict = {}
        ):
        super(SentimentModel, self).__init__()

        config = Tokenizer(setting).Config

        (
            self.bert,
            self.dropout,
            self.classifier,
        ) = (
            BertModel.from_pretrained(encodeing),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):

        return self.classifier(
            self.dropout(
                self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids).pooler_output
            )
        )
        