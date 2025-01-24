#from melo.api import TTS
import warnings
import os
from tqdm import tqdm

import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

import onnxruntime
import pnnx

import utils
import commons

from torch import _weight_norm
from torch.nn import Parameter

import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

import onnxruntime
import pnnx

from models import SynthesizerTrn
from split_utils import split_sentence
from mel_processing import spectrogram_torch, spectrogram_torch_conv
from download_utils import load_or_download_config, load_or_download_model


def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        #分割句子
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts


language = 'ZH'
device="cpu"

use_hf=True
config_path=None
ckpt_path=None

hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

num_languages = hps.num_languages
num_tones = hps.num_tones
symbols = hps.symbols

melotts = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    num_tones=num_tones,
    num_languages=num_languages,
    **hps.model,
)

speaker_ids = hps.data.spk2id

_ = melotts.eval()
symbol_to_id = {s: i for i, s in enumerate(symbols)}
    
# load state_dict
checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
melotts.load_state_dict(checkpoint_dict['model'], strict=True)




ptmodel= melotts
dec = ptmodel.dec
dec_x = [torch.randn((1,192,255))]
torch.jit.trace(dec, dec_x)
pnnx.export(dec, "dec.pt", dec_x)



text="这是为了给一个正常的输入以方便采集输入矩阵"
spkr = speaker_ids[list(speaker_ids.keys())[0]]
output_path = "out.wav"
speed = 0.8
speaker_id = spkr;
quiet = False
sdp_ratio=0.2;
noise_scale=0.6;
noise_scale_w=0.8;
speed=1.0;

texts = split_sentences_into_pieces(text, language, quiet)

tx = tqdm(texts)

t = next(iter(tx))

if language in ['EN', 'ZH_MIX_EN']:
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)

bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, hps, device, symbol_to_id)

#add for pnnx
x_tst = phones.to(device).unsqueeze(0)
x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
speakers = torch.LongTensor([speaker_id]).to(device)
tones = tones.to(device).unsqueeze(0)
lang_ids = lang_ids.to(device).unsqueeze(0)
bert = bert.to(device).unsqueeze(0)
ja_bert = ja_bert.to(device).unsqueeze(0)

t_noise_scale = torch.tensor([noise_scale]);
t_noise_scale_w = torch.tensor([noise_scale_w]);
t_length_scale = torch.tensor([1. / speed]);
t_sdp_ratio = torch.tensor([sdp_ratio]);

x = (x_tst, 
     x_tst_lengths,
     speakers,
     tones,
     lang_ids,
     bert,
     ja_bert,
     #t_noise_scale,
     #t_length_scale,
     #t_noise_scale_w,
     #t_sdp_ratio
     )
print(x)


#opt_model = pnnx.export(ptmodel, "melo.pt", x)
emb_g = ptmodel.emb_g
emb_g_x = speakers;
torch.jit.trace(emb_g, emb_g_x)
pnnx.export(emb_g, "emb_g.pt", emb_g_x)

atf_emb_g_g = emb_g(emb_g_x).unsqueeze(-1)


#g = self.emb_g(sid).unsqueeze(-1)

# x, m_p, logs_p, x_mask = self.enc_p(
#            x, x_lengths, tone, language, bert, ja_bert, g=g_p
#        )
enc_p = ptmodel.enc_p
enc_p_x = (x_tst, 
     x_tst_lengths,
     tones,
     lang_ids,
     bert,
     ja_bert, atf_emb_g_g)
torch.jit.trace(enc_p, enc_p_x)
pnnx.export(enc_p, "enc_p.pt", enc_p_x)

#torch.jit.trace(ptmodel, x).save("melojit.pt")
torch.jit.trace(ptmodel, x)
aft_enc_p_x, aft_enc_p_m_p, aft_enc_p_logs_p, aft_enc_p_x_mask = ptmodel.enc_p(
            x_tst, x_tst_lengths, tones, lang_ids, bert, ja_bert, atf_emb_g_g)

#        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
#            sdp_ratio
#        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)


sdp = ptmodel.sdp
sdp_x = (
    aft_enc_p_x, #x
    aft_enc_p_x_mask.to(dtype=torch.int64), #x_mask 
    atf_emb_g_g,  # w
    #reverse=True, 
    t_noise_scale_w)
torch.jit.trace(sdp, sdp_x)
pnnx.export(sdp, "sdp.pt", sdp_x)

aft_sdp_logw = ptmodel.sdp(aft_enc_p_x, aft_enc_p_x_mask.to(dtype=torch.int64), atf_emb_g_g, t_noise_scale_w)


dp = ptmodel.dp
dp_x = (
    aft_enc_p_x, #x
    aft_enc_p_x_mask.to(dtype=torch.int64), #x_mask 
    atf_emb_g_g)
torch.jit.trace(dp, dp_x)
pnnx.export(dp, "dp.pt", dp_x)


aft_dp_logw =  ptmodel.dp(aft_enc_p_x, aft_enc_p_x_mask.to(dtype=torch.int64), atf_emb_g_g)
atf_merge_logw = aft_sdp_logw * (sdp_ratio) + aft_dp_logw * (1 - sdp_ratio)


aft_exp_w = torch.exp(atf_merge_logw) * aft_enc_p_x_mask * t_length_scale


aft_w_ceil = torch.ceil(aft_exp_w)
aft_y_lengths = torch.clamp_min(torch.sum(aft_w_ceil, [1, 2]), 1).long()
aft_y_mask = torch.unsqueeze(commons.sequence_mask(aft_y_lengths, None), 1).to(dtype=torch.int64)
aft_attn_mask = torch.unsqueeze(aft_enc_p_x_mask, 2) * torch.unsqueeze(aft_y_mask, -1)
aft_attn = commons.generate_path(aft_w_ceil, aft_attn_mask).to(dtype=torch.float)

aft_m_p = torch.matmul(aft_attn.squeeze(1), aft_enc_p_m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
aft_logs_p = torch.matmul(aft_attn.squeeze(1), aft_enc_p_logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

aft_z_p = aft_m_p + torch.randn_like(aft_m_p) * torch.exp(aft_logs_p) * t_noise_scale


flow = ptmodel.flow
flow_x = (
    aft_z_p, #x
    aft_y_mask, #x_mask 
    atf_emb_g_g)
torch.jit.trace(flow, flow_x)
pnnx.export(flow, "flow.pt", flow_x)

aft_z = model.model.flow(aft_z_p, aft_y_mask, atf_emb_g_g)

max_len = None
pre_dec_x = (aft_z * aft_y_mask)[:, :, :max_len]

dec = ptmodel.dec
dec_x = (
    pre_dec_x, #x
    atf_emb_g_g)
torch.jit.trace(dec, dec_x)
pnnx.export(dec, "dec.pt", dec_x)


output = model.model.dec((aft_z * aft_y_mask)[:, :, :max_len], atf_emb_g_g)
#model.tts_to_file(text, spkr, output_path, speed=speed)

input_names = ['pre_dec_x', 'atf_emb_g_g']
output_names = ['audio']
                
torch.onnx.export(dec, dec_x, 'dec.onnx', input_names=input_names, output_names=output_names, verbose='True', opset_version=12)


dec = ptmodel.dec
dec.remove_weight_norm();
dec_x = [torch.randn((1,192,255)), torch.randn((1,256,1))]
torch.jit.trace(dec, dec_x)
pnnx.export(dec, "dec.pt", dec_x)



audio1 = model.model.forward(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        #noise_scale,
                        #1. / speed,
                        #noise_scale_w,
                        #sdp_ratio,
                    )[0][0, 0].data.cpu().float().numpy()

audio = model.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()