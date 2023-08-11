import os
import re
import math
import torch
import numpy as np
from torch import nn
import tensorflow as tf
import torch.utils.checkpoint
from transformers import logging
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.pytorch_utils import ( 
    apply_chunking_to_forward, 
    find_pruneable_heads_and_indices, 
    prune_linear_layer 
) 
# off logger
logging.set_verbosity_error()

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    
    tf_path = os.path.abspath(tf_checkpoint_path)

    for name, array in zip(names:= [name for name, shape in tf.train.list_variables(tf_path)], [tf.train.load_variable(tf_path, name) for _ in names ]):
        name = name.split("/")

        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            continue

        pointer = model

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")

        elif m_name == "kernel":
            array = np.transpose(array)

        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        pointer.data = torch.from_numpy(array)
    return model

@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertConfig(PretrainedConfig):
    
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        (
            self.vocab_size , 
            self.hidden_size , 
            self.num_hidden_layers , 
            self.num_attention_heads , 
            self.hidden_act , 
            self.intermediate_size , 
            self.hidden_dropout_prob , 
            self.attention_probs_dropout_prob , 
            self.max_position_embeddings , 
            self.type_vocab_size , 
            self.initializer_range , 
            self.layer_norm_eps , 
            self.position_embedding_type , 
            self.use_cache , 
            self.classifier_dropout , 
        ) = (

            vocab_size , 
            hidden_size , 
            num_hidden_layers , 
            num_attention_heads , 
            hidden_act , 
            intermediate_size , 
            hidden_dropout_prob , 
            attention_probs_dropout_prob , 
            max_position_embeddings , 
            type_vocab_size , 
            initializer_range , 
            layer_norm_eps , 
            position_embedding_type , 
            use_cache , 
            classifier_dropout , 

        )

class BertEmbeddings(nn.Module):
    def __init__(self, config : BertConfig):
        super().__init__()
        (
            self.word_embeddings , 
            self.position_embeddings , 
            self.token_type_embeddings , 
            self.LayerNorm , 
            self.dropout , 
            self.position_embedding_type , 
        ) = (
            nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) , 
            nn.Embedding(config.max_position_embeddings, config.hidden_size) , 
            nn.Embedding(config.type_vocab_size, config.hidden_size) , 
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) , 
            nn.Dropout(config.hidden_dropout_prob) , 
            getattr(config, "position_embedding_type", "absolute") , 
        )

        # run function 
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length] if position_ids is None else position_ids

        embeddings = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds + self.token_type_embeddings(
                ( 
                    self.token_type_ids[:, :seq_length].expand(input_shape[0], seq_length) 
                        if hasattr(self, "token_type_ids") 
                        else 
                    torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device) 
                ) if token_type_ids is None else token_type_ids
            )
        
        if self.position_embedding_type == "absolute":
            embeddings += self.position_embeddings( self.position_ids[:, past_key_values_length : seq_length + past_key_values_length] if position_ids is None else position_ids )
  
        return self.dropout(
            self.LayerNorm(embeddings)
        )

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        return x.view( 
                x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            ).permute(0, 2, 1, 3)
  
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer , value_layer , attention_mask = past_key_value[0] , past_key_value[1] , encoder_attention_mask

        elif is_cross_attention:
            key_layer , value_layer , attention_mask = self.transpose_for_scores(self.key(encoder_hidden_states)) , self.transpose_for_scores(self.value(encoder_hidden_states)) , encoder_attention_mask

        elif past_key_value is not None:
            key_layer , value_layer = torch.cat([past_key_value[0], self.transpose_for_scores(self.key(hidden_states))], dim=2) , torch.cat([past_key_value[1], self.transpose_for_scores(self.value(hidden_states))], dim=2)

        else:
            key_layer , value_layer = self.transpose_for_scores(self.key(hidden_states)) , self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]

            positional_embedding = self.distance_embedding(
                (torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(-1, 1) if use_cache else torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)) - torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1) + self.max_position_embeddings - 1
            ).to(dtype=query_layer.dtype)


            if self.position_embedding_type == "relative_key":
                attention_scores = attention_scores + torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)

            elif self.position_embedding_type == "relative_key_query":
                attention_scores = attention_scores + torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding) + torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # error 

        attention_probs = self.dropout( nn.functional.softmax(attention_scores, dim=-1) )

        if head_mask is not None:
            attention_probs = attention_probs * head_mask


        context_layer = torch.matmul(attention_probs, value_layer).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.dense ,
            self.LayerNorm ,
            self.dropout , 
        ) = (
            nn.Linear(config.hidden_size, config.hidden_size) , 
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) , 
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout( self.dense(hidden_states) ) + input_tensor)

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        (
            self.self , 
            self.output , 
            self.pruned_heads , 
        ) = (
            BertSelfAttention(config, position_embedding_type=position_embedding_type) , 
            BertSelfOutput(config) , 
            set() , 
        )

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
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
            self.LayerNorm ,
            self.dropout , 
        ) = (
            nn.Linear(config.intermediate_size, config.hidden_size) , 
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) , 
            nn.Dropout(config.hidden_dropout_prob)
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
            self.is_decoder , 
            self.add_cross_attention , 
        ) = (
            config.chunk_size_feed_forward , 
            1 , 
            BertAttention(config) , 
            config.is_decoder , 
            config.add_cross_attention , 
        )

        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value= past_key_value[:2] if past_key_value is not None else None ,
        )

        outputs = self_attention_outputs[1:-1] if self.is_decoder else self_attention_outputs[1:] 

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_outputs = self.crossattention(
                self_attention_outputs[0],
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value[-2:] if past_key_value is not None else None,
                output_attentions,
            )
            self_attention_outputs[0] , outputs , present_key_value = cross_attention_outputs[0] , outputs + cross_attention_outputs[1:-1] , self_attention_outputs[-1] + cross_attention_outputs[-1]

        return outputs + (present_key_value,) if self.is_decoder else (apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, self_attention_outputs[0]),) + outputs

    def feed_forward_chunk(self, attention_output):
        return self.output(self.intermediate(attention_output) , attention_output)

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.config , 
            self.layer , 
            self.gradient_checkpointing ,
        ) = (
            config , 
            nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)]) , 
            False , 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        (
            all_hidden_states , 
            all_self_attentions , 
            all_cross_attentions , 
            use_cache , 
        ) = (
            () if output_hidden_states else None , 
            () if output_attentions else None , 
            () if output_attentions and self.config.add_cross_attention else None  , 
            False if self.gradient_checkpointing and self.training and use_cache else use_cache  , 
        )
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            (
                all_hidden_states , 
                layer_head_mask , 
                past_key_value , 
            ) = (
                all_hidden_states + (hidden_states,) if output_hidden_states else all_hidden_states , 
                head_mask[i] if head_mask is not None else None , 
                past_key_values[i] if past_key_values is not None else None , 
            )

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)
                return custom_forward
            
            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            ) if self.gradient_checkpointing and self.training else layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )


            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.dense , 
            self.activation , 
        ) = (
            nn.Linear(config.hidden_size, config.hidden_size) , 
            nn.Tanh() , 
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        (
            self.dense , 
            self.transform_act_fn , 
            self.LayerNorm , 
        ) = (
            nn.Linear(config.hidden_size, config.hidden_size) , 
            ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act , 
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) , 
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.transform_act_fn(self.dense(hidden_states)))

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        (
            self.transform , 
            self.decoder , 
            self.bias , 
        ) = (
            BertPredictionHeadTransform(config) , 
            nn.Linear(config.hidden_size, config.vocab_size, bias=False) , 
            nn.Parameter(torch.zeros(config.vocab_size))
        )

        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        return self.decoder(self.transform(hidden_states))

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return self.predictions(sequence_output)

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        return self.seq_relationship(pooled_output)

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        (
            self.predictions ,
            self.seq_relationship ,
        ) = (
            BertLMPredictionHead(config) ,
            nn.Linear(config.hidden_size, 2) ,
        )

    def forward(self, sequence_output, pooled_output):
        return self.predictions(sequence_output) , self.seq_relationship(pooled_output)

# clanned output
class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

@dataclass
class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        (
            self.config , 
            self.embeddings , 
            self.encoder , 
            self.pooler , 
        ) = (
            config , 
            BertEmbeddings(config) , 
            BertEncoder(config) , 
            BertPooler(config) if add_pooling_layer else None , 
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        (
            output_attentions , 
            output_hidden_states , 
            return_dict , 
            use_cache , 
        ) = (
            output_attentions if output_attentions is not None else self.config.output_attentions , 
            (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states) , 
            return_dict if return_dict is not None else self.config.use_return_dict , 
            use_cache if use_cache is not None else self.config.use_cache if self.config.is_decoder else False , 
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        elif input_ids is not None:
            input_shape = input_ids.size()

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length), device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=( self.embeddings.token_type_ids[:, :seq_length].expand(batch_size, seq_length) if hasattr(self.embeddings, "token_type_ids") else torch.zeros(input_shape, dtype=torch.long, device=device) ) if token_type_ids is None else token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=self.get_extended_attention_mask(torch.ones(((batch_size, seq_length + past_key_values_length)), device=device) if attention_mask is None else attention_mask , input_shape),
            head_mask=self.get_head_mask(head_mask, self.config.num_hidden_layers),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    