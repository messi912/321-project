import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Character-level encoder-decoder with attention.
# The encoder reads "<to:target> <lang> chars <lang> chars ..." and the decoder writes the
# target form one character at a time, attending over the source characters at each step.
# Attention lets it learn character alignments (e.g. Hawaiian k -> Maori t) on its own,
# which is what the mapped baseline does by hand.


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_id, embed_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.pad_id = pad_id
        # Sources and targets use the same characters, so they share one embedding table
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        # Each decoder step also sees the previous step's attention context ("input feeding")
        self.decoder = nn.GRUCell(embed_dim + 2 * hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim + 2 * hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, source):
        """source: (batch, source_len) IDs -> encoder outputs, padding mask, initial decoder state."""
        mask = source != self.pad_id
        lengths = mask.sum(dim=1).cpu()
        embedded = self.dropout(self.embedding(source))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.encoder(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=source.size(1))

        # Start the decoder from the average of the (non-padding) encoder outputs
        mean = (outputs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        hidden = torch.tanh(self.bridge(mean))
        context = torch.zeros_like(mean)
        return outputs, mask, hidden, context

    def step(self, previous, hidden, context, outputs, mask):
        """One decoder step: previous character IDs (batch,) -> next-character logits."""
        embedded = self.dropout(self.embedding(previous))
        hidden = self.decoder(torch.cat([embedded, context], dim=-1), hidden)

        scores = torch.bmm(outputs, self.attention(hidden).unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)

        logits = self.output(self.dropout(torch.cat([hidden, context], dim=-1)))
        return logits, hidden, context, weights

    def forward(self, source, target):
        """Teacher forcing: feed the true previous character, return logits for target[:, 1:]."""
        outputs, mask, hidden, context = self.encode(source)
        all_logits = []
        for t in range(target.size(1) - 1):
            logits, hidden, context, _ = self.step(target[:, t], hidden, context, outputs, mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, source, start_id, end_id, max_len=30):
        """Pick the most likely character at each step. Returns (batch, <=max_len) IDs without <start>."""
        outputs, mask, hidden, context = self.encode(source)
        previous = torch.full((source.size(0),), start_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        predictions = []
        for _ in range(max_len):
            logits, hidden, context, _ = self.step(previous, hidden, context, outputs, mask)
            previous = logits.argmax(dim=-1)
            predictions.append(previous)
            finished |= previous == end_id
            if finished.all():
                break
        return torch.stack(predictions, dim=1)
