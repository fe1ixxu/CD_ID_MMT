# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,   
)

from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)

from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)
from fairseq.modules.linear import Linear

class VariationalTransformerEncoder(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

        self.output_layer = Linear(
                embed_tokens.embedding_dim,
                len(dictionary),
                bias=False,
                init_model_on_gpu=cfg.init_model_on_gpu,
            )
        nn.init.normal_(
            self.output_layer.weight, mean=0, std=embed_tokens.embedding_dim**-0.5
        )

        self.num_lang = len(args.langs)
        self.variational_module = VariationalModule(model_size=embed_tokens.embedding_dim, latent_size=512, drop_rate=cfg.dropout)
        self.lang_embeds = Language_Embedding(lang_num=self.num_lang, model_size=embed_tokens.embedding_dim)

        self.dictionary = dictionary
        

    def build_encoder_layer(self, args, is_moe_layer=False):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args), is_moe_layer=is_moe_layer
        )
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_langs=None,
        tgt_langs=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, src_langs, tgt_langs
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        src_langs = None,
        tgt_langs = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        device = src_tokens.device
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        results = {
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "fc_results": [],  # List[T x B x C]
            "src_tokens": [src_lengths],
            "src_lengths": [],
        }

        if return_all_hiddens:
            results["encoder_states"].append(x)

        # encoder layers
        loss_keys = ["moe_gate_loss", "clsr_gate_loss_num", "clsr_gate_loss_denom"]
        for key in loss_keys:
            results[key] = []
        dropout_probs = torch.empty(len(self.layers)).uniform_()
        for i, layer in enumerate(self.layers):
            passed_src_tokens = (
                src_tokens if self.cfg.pass_tokens_transformer_layer else None
            )
            lr, l_aux_i = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                tokens=passed_src_tokens,
            )
            if isinstance(lr, tuple) and len(lr) == 2:
                tmp_x, fc_result = lr
            else:
                tmp_x = lr
                fc_result = None
            moe_layerdrop = (
                getattr(layer, "is_moe_layer", False) or not self.cfg.moe_layerdrop_only
            )
            if (
                self.training
                and (dropout_probs[i] < self.encoder_layerdrop)
                and moe_layerdrop
            ):
                x = x + tmp_x * 0
                if l_aux_i is not None:
                    for k, v in l_aux_i.items():
                        l_aux_i[k] = v * 0
            else:
                x = tmp_x
            if return_all_hiddens and not torch.jit.is_scripting():
                results["encoder_states"].append(x)
                results["fc_results"].append(fc_result)
            for key in loss_keys:
                results[key].append((l_aux_i or {}).get(key, None))

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # get variational out
        x, variational_loss = self.variational_module(x)

        # compute lang class loss
        true_src_langs = torch.tensor([len(self.dictionary) - 2 - s for s in src_langs]).to(device)
        fake_src_langs = (true_src_langs + torch.randint(0, self.num_lang, (1,)).to(device)) % self.num_lang
        labels = torch.cat([torch.ones_like(true_src_langs) - 0.1, torch.zeros_like(fake_src_langs) + 0.1], dim=-1).type(x.dtype)
        all_langs = torch.cat([true_src_langs, fake_src_langs], dim=-1)
        lang_embeddings = self.lang_embeds(all_langs) #[2*bz, dim]

        mean_encoder_out = torch.mean(x, dim=0) #[bz, dim]
        mean_encoder_out = torch.cat([mean_encoder_out]*2, dim=0) #[2*bz, dim]
        sim = torch.cosine_similarity(mean_encoder_out.detach(), lang_embeddings)
        lang_class_loss = torch.nn.functional.mse_loss(sim, labels)

        # inject target lang embed to encoder out:
        tgt_langs = torch.tensor([len(self.dictionary) - 2 - s for s in tgt_langs]).to(device)
        tgt_lang_embed = self.lang_embeds(tgt_langs)
        x = x + tgt_lang_embed.detach()

        results["encoder_out"] = [x]  # T x B x C
        results["variational_loss"] = [variational_loss]
        results["lang_class_loss"] = [lang_class_loss]

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return results

class VariationalModule(nn.Module):
    def __init__(self, model_size, latent_size, drop_rate):
        super(VariationalModule, self).__init__()
        self.fc_mu = nn.Sequential(
            nn.Linear(model_size, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, model_size),
            FairseqDropout(
                drop_rate,
                module_name=self.__class__.__name__,
            )
        )
        self.fc_var = nn.Sequential(
            nn.Linear(model_size, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, model_size),
            FairseqDropout(
                drop_rate,
                module_name=self.__class__.__name__,
            )
        )

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        z = self.reparameterize(mu, log_var)
        loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())
        return z, loss
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_random_latent(self, sample_shape, gpu):
        z = torch.randn(sample_shape)
        z = z.to(gpu)
        return z

class Language_Embedding(nn.Module):
    def __init__(self, lang_num, model_size):
        super(Language_Embedding, self).__init__()
        self.embeddings = nn.Embedding(lang_num, model_size)
        nn.init.normal_(
            self.embeddings.weight, mean=0, std=model_size**-0.5
        )
    def forward(self, x):
        return self.embeddings(x)

@register_model("transformer_variational")
class VariationalTransformerModel(TransformerModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """
    def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return VariationalTransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        step = None,
        encoder_out = None,
        layer_input = None,
        src_langs = None,
        tgt_langs = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if step is None:
            raise NotImplementedError("Not go through here!")
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            return decoder_out
        elif step == "encoder":
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, src_langs=src_langs, tgt_langs=tgt_langs,
            )
            return encoder_out
        elif step == "decoder":
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            )
            return decoder_out
        elif step == "encoder_proj_layer":
            return self.encoder.output_layer(layer_input)
        elif step == "decoder_proj_layer":
            return self.decoder.output_layer(layer_input)
        else:
            ValueError("No such step, it can only be None, encoder, and decoder")

# architectures


@register_model_architecture("transformer_variational", "transformer_variational")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)