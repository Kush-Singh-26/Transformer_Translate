from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from .model_def import BuildTransformer


app = FastAPI(title="Hindi-English Translator API")
model = None
tokenizer = None
device = torch.device("cpu") 


class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.on_event("startup")
def load_assets():

    global model, tokenizer, device

    model_file = hf_hub_download(repo_id="Kush26/Transformer_Translation", filename="model.pth")
    tokenizer_file = hf_hub_download(repo_id="Kush26/Transformer_Translation", filename="hindi-english_bpe_tokenizer.json")
    
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()

    config = {
        "d_model": 256,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "max_seq_len": 512,
    }

    model = BuildTransformer(
        src_vocab_size=vocab_size,
        trg_vocab_size=vocab_size,
        src_seq_len=config["max_seq_len"],
        trg_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    ).to(device)

    # 5. Load the trained weights
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode

    print("✅ Model and Tokenizer loaded successfully!")


def greedy_decode(sentence: str, max_len=100):
    PAD_token = tokenizer.token_to_id('[PAD]')
    
    model.eval()
    src_ids = [tokenizer.token_to_id('[SOS]')] + tokenizer.encode(sentence).ids + [tokenizer.token_to_id('[EOS]')]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor != PAD_token).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    tgt_tokens = [tokenizer.token_to_id('[SOS]')]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        trg_mask_padding = (tgt_tensor != PAD_token).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones(1, tgt_tensor.size(1), tgt_tensor.size(1), device=device)).bool()
        trg_mask = trg_mask_padding & subsequent_mask

        with torch.no_grad():
            decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, trg_mask)
            logits = model.project(decoder_output)
        
        pred_token = logits.argmax(dim=-1)[0, -1].item()
        tgt_tokens.append(pred_token)
        
        if pred_token == tokenizer.token_to_id('[EOS]'):
            break
            
    return tokenizer.decode(tgt_tokens, skip_special_tokens=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hindi-English Translator API"}

@app.post("/translate/greedy", response_model=TranslationResponse)
def translate_greedy_endpoint(request: TranslationRequest):

    translated_text = greedy_decode(request.text)
    return {"translated_text": translated_text}