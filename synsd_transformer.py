"""
Main model script for the SynSD pipeline

Author: Amandeep Singh Hira
email: ahira1@ualberta.ca
Date: September 2025

References: 

ualbertaIGEM. (2025). Ashbloom https://2025.igem.wiki/ualberta.

test.svg: https://www.svgrepo.com/svg/530662/ribosome

The pandas development team. (2020). pandas [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.3509134

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Oliphant, T. E. (2020). NumPy [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.4147899

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... Chintala, S. (2019). PyTorch [Computer software]. https://pytorch.org/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). scikit-learn: Machine learning in Python [Computer software]. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org/

Streamlit Inc. (2023). Streamlit [Computer software]. https://streamlit.io/

Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., ... de Hoon, M. J. L. (2009). Biopython [Computer software]. Bioinformatics, 25(11), 1422–1423. https://biopython.org/

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment [Computer software]. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. (2021). Seaborn [Computer software]. https://seaborn.pydata.org/

Lorenz, R., Bernhart, S. H., Höner zu Siederdissen, C., Tafer, H., Flamm, C., Stadler, P. F., & Hofacker, I. L. (2011). ViennaRNA Package [Computer software]. https://www.tbi.univie.ac.at/RNA/

BioRender. (n.d.). BioRender [Computer software]. https://biorender.com/


Description: 
A ribosome binding site (RBS) refers to a brief segment of mRNA that plays a crucial role in attracting the ribosome to start the translation process, thus influencing the efficiency of protein production. For microbial synthetic biology, where meticulous regulation of gene expression is essential, the precise prediction and design of RBS sequences are vital.
Current RBS prediction models predominantly rely on RNA thermodynamics. The most prevalent method is created by the Salis Lab called Denovo DNA [1]. The Denovo DNA estimates RBS sequences by assessing the minimum folding energy of the mRNA alongside the ribosome binding energy, establishing a thermodynamic basis for RBS design [1].
In contrast, we developed a novel deep learning model to predict ribosome binding sites (RBS) and spacer sequences from mRNA contexts. Inspired by natural sequence-to-function mappings, we employ a Transformer encoder–decoder architecture, capable of learning long-range dependencies in RNA. The goal is to enable rational RBS+spacer design for microbial synthetic biology.
The model is coupled with the extraction and analysis code to form a pipeline for easier user interaction. The pipeline takes an annotated genome as an input and extracts out all the ribosome binding sequences and spacer sequences with their respective mRNA sequences. The resulting sequences are used to train the model for sequence prediction. 
The model is validated by calculating the minimum folding energies of the first 25 base pairs of the RNA and the ribosome-RNA sequence binding affinity. As the paper by Chen, Yi-Lan, and Jin-Der Wen elucidates mRNA–ribosome complexes that use less favorable / more structured RBS tend to be disfavored during initiation, via kinetic discrimination [2].

1. Reis, A.C. & Salis, H.M. (2020). An automated model test system for systematic development and improvement of gene expression models. ACS Synthetic Biology, 9(11), 3145-3156.
2. Chen, Yi-Lan, and Jin-Der Wen. "Translation initiation site of mRNA is selected through dynamic interaction with the ribosome." Proceedings of the National Academy of Sciences 119.22 (2022): e2118099119.



"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time


log_file = "training_log.txt" # log file for GUI

# Tokenization / Vocab
def create_bio_vocab():
    """
    RNA-normalized vocab (T is mapped to U during indexing).
    """
    vocab = {
        'A': 0, 'G': 1, 'C': 2, 'U': 3,            # RNA bases
        '<PAD>': 4,
        '<SOS>': 5,
        '<EOS>': 6,
        'N': 7                                     # ambiguous
    }
    idx_to_token = {idx: tok for tok, idx in vocab.items()}
    return vocab, idx_to_token

def sequence_to_indices(sequence, vocab, max_len=None):
    """Convert DNA/RNA string to RNA indices. Maps T->U, preserves N."""
    if pd.isna(sequence) or sequence == '':
        return []
    seq = sequence.upper().replace('T', 'U')  # normalize to RNA
    indices = [vocab.get(ch, vocab['N']) for ch in seq]
    if max_len is not None:
        if len(indices) < max_len:
            indices = indices + [vocab['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
    return indices

def indices_to_sequence(indices, idx_to_token):
    """Convert list of indices back to sequence (keeps 'N', drops special tokens)."""
    out = []
    for idx in indices:
        tok = idx_to_token[int(idx)]
        if tok in ('<PAD>', '<SOS>', '<EOS>'):
            continue
        out.append(tok)
    return ''.join(out)


# Data


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['RBS_sequence', 'spacer_sequence', 'mRNA_sequence'])
    df = df[(df['mRNA_sequence'].str.len() > 0) &
            (df['RBS_sequence'].str.len() > 0)]
    return df

class RBSSpacerDataset(Dataset):
    def __init__(self, df, vocab, idx_to_token, mrna_max_len=200, target_max_len=50):
        self.vocab = vocab
        self.idx_to_token = idx_to_token
        self.mrna_max_len = mrna_max_len
        self.target_max_len = target_max_len
        self.PAD = vocab['<PAD>']
        self.SOS = vocab['<SOS>']
        self.EOS = vocab['<EOS>']

        self.data = []
        for _, row in df.iterrows():
            mrna_seq = row['mRNA_sequence']
            rbs_seq = row['RBS_sequence']
            spacer_seq = row['spacer_sequence']

            target_seq = (rbs_seq or '') + (spacer_seq or '')
            mrna_indices = sequence_to_indices(mrna_seq, vocab, mrna_max_len)

            tgt = [self.SOS] + sequence_to_indices(target_seq, vocab) + [self.EOS]
            if len(tgt) < target_max_len:
                tgt = tgt + [self.PAD] * (target_max_len - len(tgt))
            else:
                tgt = tgt[:target_max_len]
                tgt[-1] = self.EOS

            self.data.append({
                'mrna': torch.tensor(mrna_indices, dtype=torch.long),
                'target': torch.tensor(tgt, dtype=torch.long),
                'rbs_seq': rbs_seq,
                'spacer_seq': spacer_seq
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Transformer Blocks


class PositionalEncoding(nn.Module):
    """
    Batch-first positional encoding.
    Input/Output: [B, S, D]
    """
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)                 # [S, D]
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                            # [S, D]

    def forward(self, x):                                         # [B, S, D]
        S = x.size(1)
        return x + self.pe[:S, :].unsqueeze(0)                    # [1, S, D]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: [B,H,Sq,Dk], K: [B,H,Sk,Dk] -> scores: [B,H,Sq,Sk]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask True=keep, False=block; broadcast to [B,H,Sq,Sk] if needed
            attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, V)  # [B,H,Sq,Dk]

    def forward(self, query, key, value, mask=None):
        # query: [B,Sq,D], key: [B,Sk,D], value: [B,Sv,D] (Sv==Sk expected)
        B, Sq, _ = query.size()
        Sk = key.size(1)
        Sv = value.size(1)

        Q = self.W_q(query).view(B, Sq, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,Sq,Dk]
        K = self.W_k(key).view(B, Sk, self.num_heads, self.d_k).transpose(1, 2)    # [B,H,Sk,Dk]
        V = self.W_v(value).view(B, Sv, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,Sv,Dk]

        # Expand mask [B,1,*,*] -> [B,H,*,*] if needed
        if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.expand(-1, self.num_heads, -1, -1)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)             # [B,H,Sq,Dk]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        return self.W_o(attn_output)                                                # [B,Sq,D]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, self_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, cross_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# Full Model


class RBSSpacerTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=16, num_encoder_layers=12,
                 num_decoder_layers=12, d_ff=2048, max_seq_length=1000, dropout=0.1,
                 pad_idx=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_seq_length = max_seq_length

        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)

    # ---- masks ----
    def create_src_mask(self, src):
        # [B, 1, 1, S] True=keep
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def build_tgt_mask(self, tgt):
        # padding mask: [B,1,1,S]
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        S = tgt.size(1)
        # causal mask: [1,S,S]
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=tgt.device)).unsqueeze(0)
        # combine -> [B,1,S,S]
        return pad_mask & causal

    # ---- encode/decode ----
    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)           # [B,1,1,S_src]
        tgt_mask = self.build_tgt_mask(tgt)            # [B,1,S_tgt,S_tgt]
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        return self.output_projection(dec_out)

    @torch.no_grad()
    def generate(self, src, vocab, max_length=50):
        """
        Greedy generation (can be swapped for beam search/sampling).
        src: [B, S]
        """
        self.eval()
        device = src.device
        src_mask = self.create_src_mask(src)
        enc_out = self.encode(src, src_mask)

        SOS = vocab['<SOS>']; EOS = vocab['<EOS>']
        B = src.size(0)
        tgt = torch.full((B, 1), SOS, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            tgt_mask = self.build_tgt_mask(tgt)
            dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
            logits = self.output_projection(dec_out[:, -1, :])  # [B,V]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
            tgt = torch.cat([tgt, next_token], dim=1)
            # stop early if all sequences ended
            if torch.all(next_token.squeeze(1) == EOS):
                break
        return tgt


# Train / Eval


def train_model(model_name, model, train_loader, val_loader, vocab, num_epochs=50, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

    best_val = float('inf')

    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            mrna = batch['mrna'].to(device)
            target = batch['target'].to(device)
            dec_inp = target[:, :-1]
            dec_tgt = target[:, 1:]

            logits = model(mrna, dec_inp)                      # [B,S-1,V]
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss {loss.item():.4f}')
                with open(log_file, "a") as f:
                    f.write(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss {loss.item():.4f}\n')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                mrna = batch['mrna'].to(device)
                target = batch['target'].to(device)
                dec_inp = target[:, :-1]
                dec_tgt = target[:, 1:]
                logits = model(mrna, dec_inp)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
                val_loss += loss.item()

        avg_tr = tr_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        print(f'Epoch {epoch+1}/{num_epochs}: Train {avg_tr:.4f}  Val {avg_val:.4f}')
        with open(log_file, "a") as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}: Train {avg_tr:.4f}  Val {avg_val:.4f}\n')
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            # save hparams to ensure load-time parity
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'hparams': dict(
                    d_model=model.d_model,
                    num_heads=model.decoder_layers[0].self_attn.num_heads,
                    num_encoder_layers=len(model.encoder_layers),
                    num_decoder_layers=len(model.decoder_layers),
                    d_ff=model.decoder_layers[0].ff.fc1.out_features,
                    dropout=model.dropout.p if isinstance(model.dropout, nn.Dropout) else 0.1,
                    max_seq_length=model.max_seq_length,
                    pad_idx=model.pad_idx
                ),
                'val_loss': avg_val
            }, model_name)
            print('  New best model saved!')
            with open(log_file, "a") as f:
                f.write('  New best model saved!\n')

        

def main_train(model_name='best_rbs_spacer_model.pth', dataset_csv='ecoli_train.csv', num_epochs=50, batch_size=16):
    
    # logs
    with open(log_file, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training...\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_csv}\n")
        f.write(f"Epochs: {num_epochs}, Batch size: {batch_size}\n")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(log_file, "a") as f:
        f.write(f'Using device: {device}\n')

    # Data
    df = load_data(dataset_csv)
    vocab, idx_to_token = create_bio_vocab()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = RBSSpacerDataset(train_df, vocab, idx_to_token)
    val_dataset = RBSSpacerDataset(val_df, vocab, idx_to_token)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    with open(log_file, "a") as f:
        f.write(f'Training samples: {len(train_dataset)}\n')
        f.write(f'Validation samples: {len(val_dataset)}\n')

    model = RBSSpacerTransformer(
        vocab_size=len(vocab),
        d_model=512,
        num_heads=16,
        num_encoder_layers=12,
        num_decoder_layers=12,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=1000,
        pad_idx=vocab['<PAD>']
    )

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    train_model(model_name=model_name, model=model, train_loader=train_loader, val_loader=val_loader, vocab=vocab, num_epochs=num_epochs, device=device)




# loading utilities


def load_trained_model(model_path, device='cpu'):
    ckpt = torch.load(model_path, map_location=device)
    vocab = ckpt['vocab']
    h = ckpt['hparams']
    model = RBSSpacerTransformer(
        vocab_size=len(vocab),
        d_model=h['d_model'],
        num_heads=h['num_heads'],
        num_encoder_layers=h['num_encoder_layers'],
        num_decoder_layers=h['num_decoder_layers'],
        d_ff=h['d_ff'],
        dropout=h['dropout'],
        max_seq_length=h.get('max_seq_length', 1000),
        pad_idx=h.get('pad_idx', vocab['<PAD>'])
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    idx_to_token = {idx: tok for tok, idx in vocab.items()}
    return model, vocab, idx_to_token

@torch.no_grad()
def predict_rbs_spacer(model, mrna_sequence, vocab, idx_to_token, max_length=50, device=None):
    """Generate RBS+spacer for given mRNA. Returns (rbs_guess, spacer_guess, full_generated)."""
    if device is None:
        device = next(model.parameters()).device
    mrna_indices = sequence_to_indices(mrna_sequence, vocab, max_len=200)
    mrna_tensor = torch.tensor([mrna_indices], dtype=torch.long, device=device)
    generated = model.generate(mrna_tensor, vocab, max_length=max_length)  # [1, S]
    gen_seq = indices_to_sequence(generated[0].tolist(), idx_to_token)

    # Simple heuristic split (optional; keep your downstream logic as needed)
    if len(gen_seq) > 15:
        rbs_seq = gen_seq[:10]
        spacer_seq = gen_seq[10:]
    else:
        rbs_seq = gen_seq
        spacer_seq = ""

    return rbs_seq, spacer_seq, gen_seq