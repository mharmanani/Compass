import torch
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig


class BertBasedMILAggregatorForSequenceClassification(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        pooling_strategy="cls",
        attn_implementation="sdpa",
        num_classes=None,
        **bert_kwargs
    ):
        super().__init__()
        self.pad_token = torch.nn.Parameter(torch.randn(1, embedding_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, embedding_dim))
        self.bert = BertEncoder(
            BertConfig(
                **bert_kwargs,
                hidden_size=embedding_dim,
                attn_implementation=attn_implementation
            )
        )
        self.pooling_strategy = pooling_strategy
        self.classifier = (
            torch.nn.Linear(embedding_dim, num_classes) if num_classes else None
        )

    def forward(self, hidden_states):
        # each hidden state should be N x C

        # prepend cls token to each hidden state
        hidden_states = [torch.cat([self.cls_token, h], dim=0) for h in hidden_states]

        # pad them and create attention mask
        pad_length = max([len(h) for h in hidden_states])
        padded_hidden_states = []
        attention_masks = []
        for h in hidden_states:
            # Pad each hidden state to the maximum length
            padding = pad_length - len(h)
            padded_hidden_states.append(
                torch.cat([h, self.pad_token.expand(padding, -1)], dim=0)
            )
            attention_masks.append(
                torch.cat([torch.ones(len(h)), torch.zeros(padding)], dim=0)
            )

        padded_hidden_states = torch.stack(padded_hidden_states)
        attention_masks = (
            torch.stack(attention_masks).bool().to(padded_hidden_states.device)
        )
        bsz, seq_len = attention_masks.shape
        attention_masks = attention_masks[:, None, None, :].expand(
            bsz, 1, seq_len, seq_len
        )

        # Pass through BERT
        bert_outputs = self.bert(padded_hidden_states, attention_mask=attention_masks)[
            0
        ]
        pooled_output = self.pooling(bert_outputs)
        if self.classifier:
            return self.classifier(pooled_output)
        return pooled_output

    def pooling(self, bert_outputs):
        if self.pooling_strategy == "cls":
            return bert_outputs[:, 0]
        return bert_outputs.mean(dim=1)


class BertBasedMILAggregatorForTokenAndSequenceClassification(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        pooling_strategy="cls",
        attn_implementation="sdpa",
        num_classes=None,
        num_token_classes=None,
        disable_bert=False,
        detach_token_states_for_classification=False,
        **bert_kwargs
    ):
        super().__init__()
        self.disable_bert = disable_bert
        self.detach_token_states_for_classification = (
            detach_token_states_for_classification
        )

        self.pad_token = torch.nn.Parameter(torch.randn(1, embedding_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, embedding_dim))
        self.bert = BertEncoder(
            BertConfig(
                **bert_kwargs,
                hidden_size=embedding_dim,
                attn_implementation=attn_implementation
            )
        )
        self.pooling_strategy = pooling_strategy
        self.classifier = (
            torch.nn.Linear(embedding_dim, num_classes) if num_classes else None
        )
        self.token_classifier = (
            torch.nn.Linear(embedding_dim, num_token_classes)
            if num_token_classes
            else None
        )

    def enable_attention_output(self):
        """Switch from SDPA to eager attention so output_attentions works."""
        self.bert.config._attn_implementation = "eager"
        for layer in self.bert.layer:
            layer.attention.self._attn_implementation = "eager"

    def forward(self, hidden_states, output_attentions=False):
        # each hidden state should be N x C
        sequence_lengths = [len(h) for h in hidden_states]

        # prepend cls token to each hidden state
        hidden_states = [torch.cat([self.cls_token, h], dim=0) for h in hidden_states]

        # pad them and create attention mask
        pad_length = max([len(h) for h in hidden_states])
        padded_hidden_states = []
        attention_masks = []
        for h in hidden_states:
            # Pad each hidden state to the maximum length
            padding = pad_length - len(h)
            padded_hidden_states.append(
                torch.cat([h, self.pad_token.expand(padding, -1)], dim=0)
            )
            attention_masks.append(
                torch.cat([torch.ones(len(h)), torch.zeros(padding)], dim=0)
            )

        padded_hidden_states = torch.stack(padded_hidden_states)
        attention_masks = (
            torch.stack(attention_masks).bool().to(padded_hidden_states.device)
        )
        bsz, seq_len = attention_masks.shape
        attention_masks = attention_masks[:, None, None, :].expand(
            bsz, 1, seq_len, seq_len
        )

        # Pass through BERT
        attentions = None
        if not self.disable_bert:
            bert_result = self.bert(
                padded_hidden_states, attention_mask=attention_masks,
                output_attentions=output_attentions, return_dict=True,
            )
            bert_outputs = bert_result.last_hidden_state
            if output_attentions and bert_result.attentions is not None:
                attentions = bert_result.attentions  # tuple of (B, H, S, S)
        else:
            bert_outputs = padded_hidden_states  # identity

        bert_outputs_clstoken, bert_outputs_seqtokens = (
            bert_outputs[:, 0],
            bert_outputs[:, 1:],
        )  # (B, C), (B, N, C)
        if self.pooling_strategy != "cls":
            clstoken_hidden_state = bert_outputs.mean(1)  # take average poolin
        else:
            clstoken_hidden_state = bert_outputs_clstoken
        if self.classifier is not None:
            logits = self.classifier(clstoken_hidden_state)
        else:
            logits = None
        token_hidden_states = []
        token_logits = []
        for batch_idx in range(len(sequence_lengths)):
            token_hidden_states_i = bert_outputs_seqtokens[
                batch_idx, : sequence_lengths[batch_idx]
            ]
    
            if self.token_classifier is not None:
                if self.detach_token_states_for_classification:
                    token_logits_i = self.token_classifier(token_hidden_states_i.detach())
                else:
                    token_logits_i = self.token_classifier(token_hidden_states_i)
            else:
                token_logits_i = None
            token_hidden_states.append(token_hidden_states_i)
            token_logits.append(token_logits_i)

        result = {
            "logits": logits,
            "clstoken_logits": logits,
            "clstoken_hidden_state": clstoken_hidden_state,
            "token_hidden_states": token_hidden_states,
            "token_logits": token_logits,
        }
        if output_attentions and attentions is not None:
            result["attentions"] = attentions  # tuple of (B, H, S, S)
        return result
