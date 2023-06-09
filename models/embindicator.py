import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartPretrainedModel, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import BartConfig, BartTokenizerFast
from utils.config import *
from utils.utils import to_list
from .modules import BartModel


MODEL_CLASSES = {
    'bart': (BartConfig, BartModel, BartPretrainedModel, BartTokenizerFast)
}

model_class, config_class, pretrained_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


class GenerationModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.indicator_embs = nn.Embedding(2, config.hidden_size)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        indicator_ids=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        estep=False,
        training=True,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if training:
            if labels is not None:
                if decoder_input_ids is None:
                    decoder_input_ids = shift_tokens_right(
                        labels, self.config.pad_token_id, self.config.decoder_start_token_id
                    )
            
            if inputs_embeds is None and encoder_outputs is None: # when encoder_outputs is None, this forward is for decoding
                inputs_embeds = self.model.shared(input_ids) # (bsz, slen, hsz)
                indicator_embeds = self.indicator_embs(indicator_ids) # (bsz, slen, hsz)
                inputs_embeds = inputs_embeds + indicator_embeds

            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias # (bsz, sumlen, n_vocab)

            if estep:
                # loss_fct = CrossEntropyLoss()
                # summ_logprobs = []
                # for bidx in range(lm_logits.shape[0]):
                #     cur_lm_logits = lm_logits[bidx] # (sumlen, n_vocab)
                #     cur_loss = loss_fct(cur_lm_logits, labels[bidx])
                #     summ_logprobs.append(-cur_loss.item())

                word_logits = lm_logits.gather(dim=2, index=labels.unsqueeze(-1)).squeeze() # (bsz, sumlen)
                summ_logprobs = to_list(word_logits.sum(dim=-1)) # (bsz)
                return summ_logprobs
            else:
                masked_lm_loss = None
                if labels is not None:
                    loss_fct = CrossEntropyLoss()
                    masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

                if not return_dict:
                    output = (lm_logits,) + outputs[1:]
                    return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

                return Seq2SeqLMOutput(
                    loss=masked_lm_loss,
                    logits=lm_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                )
        else:
            inputs_embeds = self.model.shared(input_ids) # (bsz, slen, hsz)
            indicator_embeds = self.indicator_embs(indicator_ids) # (bsz, slen, hsz)
            inputs_embeds = inputs_embeds + indicator_embeds
            summary_ids = self.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, num_beams=args.num_beams,\
                    max_length=50, min_length=5, no_repeat_ngram_size=5, early_stopping=True)
            return summary_ids


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past