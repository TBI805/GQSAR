
import os
import math
import json
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from genrec.models.GRQ.grq import GRQ
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class GRQTokenizer(AbstractTokenizer):
    """GRQ Tokenizer using GRQ quantizer."""

    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'GRQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'

        super(GRQTokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100

    @property
    def n_digit(self):
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str, OpenAI=None):
        """Encodes sentence embeddings using SentenceTransformer or OpenAI API."""
        assert self.config['metadata'] == 'sentence', 'GRQ Tokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        model_path = self.config['sent_emb_model']
        self.log(f'[TOKENIZER] Using embedding model: {model_path}')
        self.log(f'[TOKENIZER] Number of sentences to encode: {len(meta_sentences)}')

        if 'text-embedding-3' in self.config['sent_emb_model']:
            # Use OpenAI API for text-embedding-3 models
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs_list = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=self.config['sent_emb_model']
                    )
                except:
                    self.log(f'[TOKENIZER] Failed to encode batch {i}, retrying with truncation...')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        n_tokens = num_tokens_from_string(sent, 'cl100k_base')
                        if n_tokens < 8192:
                            new_batch.append(sent)
                        else:
                            n_chars = 8192 / n_tokens * len(sent) - 100
                            new_batch.append(sent[:int(n_chars)])

                    responses = client.embeddings.create(input=new_batch, model=self.config['sent_emb_model'])

                for response in responses.data:
                    sent_embs_list.append(response.embedding)
            sent_embs = np.array(sent_embs_list, dtype=np.float32)
        else:
            # Use SentenceTransformer for local models (e.g., bge-large-en-v1.5, sentence-t5-base)
            self.log(f'[TOKENIZER] Loading SentenceTransformer model...')
            sent_emb_model = SentenceTransformer(model_path).to(self.config['device'])
            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
            self.log(f'[TOKENIZER] Encoding complete. Shape: {sent_embs.shape}')

        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """Returns a boolean mask indicating which items appear in training data."""
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_grq(self, sent_embs, sem_ids_path, train_mask):
        """Generates semantic IDs using GRQ."""
        device = self.config['device']
        input_dim = sent_embs.shape[1]
        hidden_dim = self.config['grq_hidden_dim']
        n_e = self.codebook_size
        e_dim = self.config['grq_e_dim']
        n_groups = self.config['grq_n_groups']

        # Calculate n_layers based on n_groups
        if n_groups > 1:
            n_layers = self.n_digit // n_groups
        else:
            n_layers = self.n_digit

        beta = self.config['grq_beta']
        lr = self.config['grq_lr']
        epochs = self.config['grq_epochs']
        batch_size = self.config['grq_batch_size']

        model = GRQ(input_dim, hidden_dim, n_e, e_dim, n_layers, n_groups, beta, config=self.config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_data = torch.tensor(sent_embs[train_mask], dtype=torch.float32).to(device)
        dataset_loader = TensorDataset(train_data)
        dataloader = DataLoader(dataset_loader, batch_size=batch_size, shuffle=True)

        self.log(f'[TOKENIZER] Training GRQ...')
        model.train()
        train_times = []

        for epoch in tqdm(range(epochs), desc='Training GRQ'):
            epoch_start_time = time.time()
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                loss, x_recon, _ = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                loss = loss + recon_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_train_time = time.time() - epoch_start_time
            train_times.append(epoch_train_time)

            if (epoch + 1) % 50 == 0:
                self.log(
                    f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_train_time:.4f}s')

        self.log(f'[TOKENIZER] Generating codes...')
        model.eval()
        all_codes = []
        chunk_size = 1024

        with torch.no_grad():
            for i in range(0, len(sent_embs), chunk_size):
                chunk = torch.tensor(sent_embs[i:i + chunk_size], dtype=torch.float32).to(device)
                codes = model.get_codes(chunk)
                all_codes.append(codes.cpu().numpy())

        pq_codes = np.concatenate(all_codes, axis=0)
        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

        model_path = sem_ids_path.replace('.sem_ids', '.pt')
        torch.save(model.state_dict(), model_path)
        self.log(f'[TOKENIZER] GRQ model saved to {model_path}')

        avg_train_time = sum(train_times) / len(train_times) if train_times else 0
        self.log(f'[TOKENIZER] Average Training Time per Epoch: {avg_train_time:.4f}s')

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            self.log(f'[TOKENIZER] GPU Memory (Max): {gpu_memory:.4f} GB')

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """Generates semantic IDs using OPQ (FAISS)."""
        import faiss

        opq_use_gpu = self.config['opq_use_gpu']
        opq_gpu_id = self.config['opq_gpu_id']
        faiss_omp_num_threads = self.config['faiss_omp_num_threads']

        if opq_use_gpu:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56

        faiss.omp_set_num_threads(faiss_omp_num_threads)

        opq_index_factory = f'OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{self.n_codebook_bits}'
        index = faiss.index_factory(sent_embs.shape[1], opq_index_factory, faiss.METRIC_INNER_PRODUCT)

        self.log(f'[TOKENIZER] Training OPQ index: {opq_index_factory}...')
        if opq_use_gpu:
            index = faiss.index_cpu_to_gpu(res, opq_gpu_id, index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if opq_use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = [bs.read(self.n_codebook_bits) for _ in range(self.n_digit)]
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """Converts semantic IDs to tokens with digit offsets."""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """Initialize the tokenizer by loading or generating semantic IDs."""
        use_opq = self.config['use_opq']
        quantizer_name = 'OPQ' if use_opq else 'GRQ'

        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}_{quantizer_name}.sem_ids'
        )

        force_regenerate_grq = not use_opq
        need_regenerate = force_regenerate_grq or not os.path.exists(sem_ids_path)

        if need_regenerate:
            if force_regenerate_grq and os.path.exists(sem_ids_path):
                self.log(f'[TOKENIZER] Force regenerating GRQ quantization...')

            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                try:
                    sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
                except ValueError as e:
                    self.log(f'[TOKENIZER] Dimension mismatch. Regenerating... Error: {e}')
                    sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            training_item_mask = self._get_items_for_training(dataset)
            if use_opq:
                self.log(f'[TOKENIZER] Using OPQ quantization...')
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
            else:
                self.log(f'[TOKENIZER] Using GRQ quantization...')
                self._generate_semantic_id_grq(sent_embs, sem_ids_path, training_item_mask)
        else:
            self.log(f'[TOKENIZER] Using cached OPQ results from {sem_ids_path}')

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """Tokenizes first n items (all losses computed in one forward pass)."""
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """Tokenizes sequence with only the last item as target."""
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenizes input example based on split type."""
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]

        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i + max_item_seq_len + 1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len + 1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """Tokenizes all datasets."""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
