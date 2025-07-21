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


class GreedyTranslationRequest(BaseModel):
    text: str

class BeamTranslationRequest(BaseModel):
    text: str
    beam_size: int = 3 


class TranslationResponse(BaseModel):
    translated_text: str


@app.on_event("startup")
def load_assets():

    global model, tokenizer, device
    local_cache_dir = "/tmp/hf_cache"
    model_file = hf_hub_download(repo_id="Kush26/Transformer_Translation", filename="model.pth", cache_dir=local_cache_dir)
    tokenizer_file = hf_hub_download(repo_id="Kush26/Transformer_Translation", filename="hindi-english_bpe_tokenizer.json", cache_dir=local_cache_dir)
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()
    config = {
        "d_model": 256, "num_layers": 6, "num_heads": 8, "d_ff": 2048, "dropout": 0.1, "max_seq_len": 512,
    }
    model = BuildTransformer(
        src_vocab_size=vocab_size, trg_vocab_size=vocab_size, src_seq_len=config["max_seq_len"], trg_seq_len=config["max_seq_len"], d_model=config["d_model"], N=config["num_layers"], h=config["num_heads"], dropout=config["dropout"], d_ff=config["d_ff"]
    ).to(device)
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    


# --- Translation Logic ---
def greedy_decode(sentence: str, max_len=100):
    # This function remains unchanged...
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

def beam_search_decode(sentence: str, beam_size: int, max_len=50):
    pad_token_id = tokenizer.token_to_id('[PAD]')
    model.eval()
    src_ids = [tokenizer.token_to_id('[SOS]')] + tokenizer.encode(sentence).ids + [tokenizer.token_to_id('[EOS]')]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_mask = (src_tensor != pad_token_id).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

    initial_beam = (torch.tensor([tokenizer.token_to_id('[SOS]')], device=device), 0.0)
    beams = [initial_beam]

    for _ in range(max_len):
        new_beams = []
        all_completed = True
        for seq, score in beams:
            if seq[-1].item() == tokenizer.token_to_id('[EOS]'):
                new_beams.append((seq, score))
                continue

            all_completed = False
            tgt_tensor = seq.unsqueeze(0)
            trg_mask_padding = (tgt_tensor != pad_token_id).unsqueeze(1).unsqueeze(2)
            subsequent_mask = torch.tril(torch.ones(1, tgt_tensor.size(1), tgt_tensor.size(1), device=device)).bool()
            trg_mask = trg_mask_padding & subsequent_mask

            with torch.no_grad():
                decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, trg_mask)
                logits = model.project(decoder_output)
            
            last_token_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_token_logits, dim=-1)
            top_log_probs, top_next_tokens = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_token = top_next_tokens[i]
                log_prob = top_log_probs[i].item()
                new_seq = torch.cat([seq, next_token.unsqueeze(0)])
                new_score = score + log_prob
                new_beams.append((new_seq, new_score))
        
        if all_completed:
            break

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq.tolist(), skip_special_tokens=True)


@app.get("/")
def read_root():
    return {"message": "Hindi-English Translator API"}

@app.post("/translate/greedy", response_model=TranslationResponse)
def translate_greedy_endpoint(request: GreedyTranslationRequest):
    translated_text = greedy_decode(request.text)
    return {"translated_text": translated_text}

@app.post("/translate/beam", response_model=TranslationResponse)
def translate_beam_endpoint(request: BeamTranslationRequest):

    translated_text = beam_search_decode(request.text, request.beam_size)
    return {"translated_text": translated_text}