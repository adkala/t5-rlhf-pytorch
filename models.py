from torch import nn, Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import torch

class SmallT5(nn.Module):
    def __init__(self):
        super().__init__()

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        # add characters to tokenizer
        self.tokenizer.add_tokens(['{', '}'])
        self.t5.resize_token_embeddings(len(self.tokenizer))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) # for generation function below
        self.t5.to(self.device)  
    
    def forward(self, **kwargs):
        x = self.t5(**kwargs)
        return x

    def generate(self, input_ids, attention_mask, max_length=512, **kwargs): # rewriting generate for RL flexibility
        decoder_input_ids = torch.Tensor([[self.t5.config.decoder_start_token_id] for _ in range(input_ids.shape[0])]).to(torch.int).to(self.device)
        i = 0
        while i < max_length:
            o = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            scores = o[0]
            token = torch.argmax(scores[:, -1, :], dim=-1)
            decoder_input_ids = torch.cat((decoder_input_ids, token.unsqueeze(-1)), dim=-1)
            
            if torch.all(token == torch.zeros(*token.shape).to(self.device)):
                break
            i += 1
        return scores, decoder_input_ids