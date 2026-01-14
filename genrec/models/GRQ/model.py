
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """Residual Block with SiLU activation."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class GRQ(AbstractModel):
    def __init__(
            self,
            config: dict,
            dataset: AbstractDataset,
            tokenizer: AbstractTokenizer
    ):
        super(GRQ, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )

        self.gpt2 = GPT2Model(gpt2config)

        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        # Prediction mode configuration
        self.prediction_group_size = config[
            'prediction_group_size'] if 'prediction_group_size' in config else self.n_pred_head
        self.use_grouped_parallel = self.prediction_group_size < self.n_pred_head
        self.context_fusion_weight = config['context_fusion_weight'] if 'context_fusion_weight' in config else 0.5

    def _map_item_tokens(self) -> torch.Tensor:
        """Maps item IDs to their semantic token sequences."""
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
               f'#Non-embedding parameters: {total_params - emb_params}\n' \
               f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """Forward pass with automatic mode selection."""
        if self.use_grouped_parallel:
            return self.forward_grouped_parallel(batch, return_loss, self.prediction_group_size)
        else:
            return self.forward_parallel(batch, return_loss)

    def forward_parallel(self, batch: dict, return_loss=True) -> torch.Tensor:
        """Full parallel prediction: predicts all digits simultaneously."""
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)

        outputs = self.gpt2(inputs_embeds=input_embs, attention_mask=batch['attention_mask'])

        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states

        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)

            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)

            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in
                            range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1) for i in
                      range(self.n_pred_head)]
            outputs.loss = torch.mean(torch.stack(losses))

        return outputs

    def forward_grouped_parallel(self, batch: dict, return_loss=True, group_size: int = 8) -> torch.Tensor:
        """Semi-autoregressive: parallel within groups, sequential between groups."""
        device = batch['input_ids'].device
        n_groups = (self.n_pred_head + group_size - 1) // group_size

        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)

        all_final_states = []
        total_loss = 0
        n_loss_terms = 0
        current_context_embs = input_embs

        for group_idx in range(n_groups):
            start_digit = group_idx * group_size
            end_digit = min((group_idx + 1) * group_size, self.n_pred_head)
            group_digits = list(range(start_digit, end_digit))

            outputs = self.gpt2(inputs_embeds=current_context_embs, attention_mask=batch['attention_mask'])
            hidden_states = outputs.last_hidden_state

            group_states = []
            for digit_idx in group_digits:
                head_output = self.pred_heads[digit_idx](hidden_states)
                group_states.append(head_output.unsqueeze(-2))
            group_final_states = torch.cat(group_states, dim=-2)
            all_final_states.append(group_final_states)

            if return_loss:
                assert 'labels' in batch, 'The batch must contain the labels.'
                label_mask = batch['labels'].view(-1) != -100

                for local_idx, digit_idx in enumerate(group_digits):
                    pred_states = group_final_states[:, :, local_idx, :]
                    pred_states_flat = pred_states.reshape(-1, self.config['n_embd'])
                    pred_states_valid = F.normalize(pred_states_flat[label_mask], dim=-1)

                    start_idx = digit_idx * self.config['codebook_size'] + 1
                    end_idx = (digit_idx + 1) * self.config['codebook_size'] + 1
                    token_emb = F.normalize(self.gpt2.wte.weight[start_idx:end_idx], dim=-1)

                    logits = torch.matmul(pred_states_valid, token_emb.T) / self.temperature
                    token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
                    local_target = token_labels[:, digit_idx] - digit_idx * self.config['codebook_size'] - 1

                    total_loss += self.loss_fct(logits, local_target)
                    n_loss_terms += 1

            # Update context for next group
            if group_idx < n_groups - 1:
                group_token_embs = []
                for digit_idx in group_digits:
                    digit_tokens = input_tokens[:, :, digit_idx]
                    digit_embs = self.gpt2.wte(digit_tokens)
                    group_token_embs.append(digit_embs)

                group_avg_embs = torch.stack(group_token_embs, dim=-1).mean(dim=-1)

                # Context fusion
                current_context_embs = current_context_embs + group_avg_embs * self.context_fusion_weight

        # Concatenate and pad group states
        max_group_size = group_size
        padded_states = []
        for gs in all_final_states:
            if gs.shape[-2] < max_group_size:
                pad_size = max_group_size - gs.shape[-2]
                padding = torch.zeros(gs.shape[0], gs.shape[1], pad_size, gs.shape[3], device=device)
                gs = torch.cat([gs, padding], dim=-2)
            padded_states.append(gs)

        final_states = torch.cat(padded_states, dim=-2)[:, :, :self.n_pred_head, :]
        outputs.final_states = final_states

        if return_loss:
            outputs.loss = total_loss / max(n_loss_terms, 1)

        return outputs

    def generate(self, batch, n_return_sequences=1):
        """Generate top-k item predictions."""
        outputs = self.forward_parallel(batch, return_loss=False)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:, 0, i, :], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)

        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
        ).mean(dim=-1)
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        return preds.unsqueeze(-1)
