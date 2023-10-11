import math , requests 
import torch , os 
from torch import nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from typing import Optional, Tuple
from pydantic import BaseModel
from typing import Any
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import BertConfig as BertConfigData
from torch.backends import cudnn
from transformers import BertTokenizer
import logging 
from bs4 import BeautifulSoup 
from torch import cuda , manual_seed

manual_seed(9999)
cuda.manual_seed(9999)
cudnn.deterministic = True

class BertEmbeddings(nn.Module):
    def __init__(self, config ):
        super().__init__()
        (
            self.word_embeddings ,
            self.position_embeddings ,
            self.LayerNorm ,
            self.dropout ,
            self.position_embedding_type ,
        ) = (
            nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) ,
            nn.Embedding(config.max_position_embeddings, config.hidden_size) ,
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) ,
            nn.Dropout(config.hidden_dropout_prob) ,
            getattr(config, "position_embedding_type", "absolute") ,
        )

        # run function
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        seq_length = input_ids.size()[1]

        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        embeddings = self.word_embeddings(input_ids)

        if self.position_embedding_type == "absolute":
            embeddings += self.position_embeddings( self.position_ids[:, past_key_values_length : seq_length + past_key_values_length] if position_ids is None else position_ids )

        return self.dropout(
            self.LayerNorm(embeddings)
        )

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # value hosting
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # model
        (
            self.query ,
            self.key ,
            self.value ,
            self.dropout
        ) = (
            nn.Linear(config.hidden_size, self.all_head_size) ,
            nn.Linear(config.hidden_size, self.all_head_size) ,
            nn.Linear(config.hidden_size, self.all_head_size) ,
            nn.Dropout(config.attention_probs_dropout_prob)
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(
                x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            ).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        context_layer = torch.matmul(
            self.dropout(
                nn.functional.softmax(
                    torch.matmul(
                        self.transpose_for_scores(
                            self.query(
                                hidden_states
                            )
                        ),
                        self.transpose_for_scores(
                            self.key(
                                hidden_states
                            )
                        ).transpose(-1, -2)
                    ) / math.sqrt(
                        self.attention_head_size
                    ) + attention_mask,
                dim=-1)
            )
        , self.transpose_for_scores(
            self.value(
                hidden_states
            )
        )).permute(0, 2, 1, 3).contiguous()

        return (context_layer.view(context_layer.size()[:-2] + (self.all_head_size,)),)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.dense ,
            self.dropout ,
            self.LayerNorm ,
        ) = (
            nn.Linear(config.hidden_size, config.hidden_size) ,
            nn.Dropout(config.hidden_dropout_prob) ,
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) ,
        )

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout( self.dense(hidden_states) ) + input_tensor)

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.self ,
            self.output ,
        ) = (
            BertSelfAttention(config) ,
            BertSelfOutput(config) ,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        self_outputs = self.self(
            hidden_states,
            attention_mask
        )
        return (self.output(self_outputs[0], hidden_states),) + self_outputs[1:]

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.dense ,
            self.intermediate_act_fn
        ) = (
            nn.Linear(config.hidden_size, config.intermediate_size) ,
            ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.dense ,
            self.dropout ,
            self.LayerNorm ,
        ) = (
            nn.Linear(config.intermediate_size, config.hidden_size) ,
            nn.Dropout(config.hidden_dropout_prob) ,
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) ,
        )

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.chunk_size_feed_forward ,
            self.seq_len_dim ,
            self.attention ,
             self.intermediate ,
             self.output
        ) = (
            config.chunk_size_feed_forward ,
            1 ,
            BertAttention(config) ,
            BertIntermediate(config) ,
            BertOutput(config)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,

        )

        return (apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, self_attention_outputs[0]),) + self_attention_outputs[1:]

    def feed_forward_chunk(self, attention_output):
        return self.output(self.intermediate(attention_output) , attention_output)

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.config ,
            self.layer ,
        ) = (
            config ,
            nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)]) ,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,

    ) -> Tuple[torch.Tensor]:

        (
            all_hidden_states ,
            all_self_attentions ,
            all_cross_attentions ,
            use_cache ,
        ) = (
            () if self.config.output_hidden_states else None ,
            () if self.config.output_attentions else None ,
            () if self.config.output_attentions and self.config.add_cross_attention else None  ,
            self.config.use_cache if self.config.is_decoder else False ,
        )

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            all_hidden_states =all_hidden_states + (hidden_states,) if self.config.output_hidden_states else all_hidden_states

            layer_outputs = layer_module(
                hidden_states,
                attention_mask
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if self.config.output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if self.config.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.Linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size) ,
            nn.Tanh() ,
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.Linear(hidden_states[:, 0])

class BertConfig(BaseModel):
    vocab_size: int = 30522,
    hidden_size: int =768,
    num_hidden_layers : int =12,
    num_attention_heads : int =12,
    intermediate_size : int =3072,
    hidden_act : str ="gelu",
    hidden_dropout_prob : float =0.1,
    attention_probs_dropout_prob : float =0.1,
    max_position_embeddings : int =512,
    type_vocab_size : int =2,
    initializer_range : float =0.02,
    layer_norm_eps : float =1e-12,
    pad_token_id : int =0,
    position_embedding_type : str ="absolute",
    use_cache : bool =True,
    classifier_dropout : Any =None,

class BertModel(nn.Module , ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config

        (
            self.embeddings ,
            self.encoder ,
            self.pooler ,
        ) = (

            BertEmbeddings(self.config) ,
            BertEncoder(self.config) ,
            BertPooler(self.config)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:

        return self.pooler(
            self.encoder(
                self.embeddings(
                    input_ids=input_ids,
                    past_key_values_length=0,
                ),
                attention_mask=self.get_extended_attention_mask( attention_mask , input_ids.size())
            )[0]
        )

class SentimentModel(nn.Module):
    def __init__(self,
            path_or_url : str = "HooshvareLab/bert-fa-base-uncased" ,
            labels : list = [] , 
            device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        ):
        super(SentimentModel, self).__init__()

        self.bert = BertModel(
            BertConfigData.from_pretrained(
                path_or_url , **{
                    'label2id': (label2id := {label: i for i, label in enumerate(labels)}),
                    'id2label': {v: k for k, v in label2id.items()},
                }
            )
        )

        self.device = device
        self.labels = labels 
        self.Tokenizer = BertTokenizer.from_pretrained(path_or_url)

    def _LoadModel(self , addres : str):
        if os.access(addres , 0):
            logging.warning("load Model : [OK]")
            self.load_state_dict(torch.load( addres , map_location=self.device ) , strict=False)

    def forward(self, input_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _Dawnload(self , id = None , url = None ):
        if url is not None :
            with open(url, "wb") as handle:
                logging.warning("Dawnload Model : [OK]")
                handle.write(
                    requests.post(
                        BeautifulSoup(
                            requests.Session().get(
                                "https://docs.google.com/uc?export=download", 
                                params = { 'id' : id }, 
                                stream = True
                            ).content, 
                            "html.parser"
                            ).find(
                                id="download-form"
                            ).get(
                                "action"
                            ), 
                            stream=True
                        ).content
                    )
                return self._LoadModel(url)
        return None

    def predict(self , text : str , indexing : bool = True , add_special_tokens=True , truncation=True , max_length=128 , return_token_type_ids=True , padding='max_length' , return_attention_mask=True , return_tensors='pt'):
        encoding = self.Tokenizer.encode_plus(
            text ,
            add_special_tokens=add_special_tokens ,
            truncation=truncation ,
            max_length=max_length ,
            return_token_type_ids=return_token_type_ids ,
            padding=padding ,
            return_attention_mask=return_attention_mask ,
            return_tensors=return_tensors
        )

        self.eval()

        with torch.no_grad():
            outputs = self.forward(
                input_ids = encoding['input_ids'].flatten().unsqueeze(0).to(self.device) , 
                attention_mask = encoding['attention_mask'].flatten().unsqueeze(0).to(self.device)
            )

        return torch.max(outputs, dim=1)[1]  if indexing else outputs

