from torch import nn 
from transformers import BertConfig
from .Model.Model.BiLinear import BertModel 

class Tokenizer():
    def __init__(self , 
                labels , 
                path = 'HooshvareLab/bert-fa-base-uncased' ,
            ) -> None:
        (
            self._path ,
            self._label 
        ) = (
            path , 
            labels
        )

    @property 
    def Config(self):
        return BertConfig.from_pretrained(
            self._path, **{
                'label2id': (label2id := {label: i for i, label in enumerate(self._label)}),
                'id2label': {v: k for k, v in label2id.items()},
            })

class SentimentModel(nn.Module):
    def __init__(self, Model , label ):
        super(SentimentModel, self).__init__()

        config = Tokenizer(label).Config

        (
            self.bert,
            self.dropout,
            self.rnn , 
            self.fc , 
        ) = (
            BertModel.from_pretrained(Model),
            nn.Dropout(config.hidden_dropout_prob),
            nn.LSTM(config.hidden_size , 256 , num_layers=2 , bidirectional=True , dropout = config.hidden_dropout_prob),
            nn.Linear(256 * 2 , config.num_labels),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):

        return self.fc(self.rnn(
            self.dropout(
                self.bert(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids).pooler_output
            )
        ))
        