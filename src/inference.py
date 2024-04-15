import torch, torchaudio, timm
import numpy as np
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from models.ast_models import ASTModel
import os, csv, argparse, wget
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list


model_path = "../checkpoint/best_audio_model.pth"
input_tdim = 512
class_num = 2
fstride = 10
tstride = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(model_path,map_location=device)
ast_mdl = ASTModel(label_dim=class_num,fstride=10,tstride=10,input_fdim=128,input_tdim=input_tdim,audioset_pretrain=False,model_size='tiny224',verbose=True)
# ast_mdl = ASTModelVis(label_dim=2,fstride=10,tstride=10,input_fdim=128, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False,model_size='tiny224',verbose=True)

print(f'[*INFO] load checkpoint: {sd}')
audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
audio_model.load_state_dict(sd,strict=True)
audio_model.eval()   

def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = [] 
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

def make_features(wav_name, mel_bins, target_length=512):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.6424894)) / (4.2628665 * 2)
    # plt.figure(figsize=(10, 4))
    # plt.imshow(fbank.t().numpy(), aspect='auto', origin='lower')
    # plt.title('Filter Bank')
    # plt.xlabel('Frames')
    # plt.ylabel('Frequency Bands')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    # plt.savefig('mel_spectrograma.png', dpi=400)
    # plt.close()

    return fbank

label_csv = '../egs/spotify_dataset/data/class_labels_indices.csv'     
labels = load_label(label_csv)
feats = make_features('../sp/7mPXeVKLdc6g7gmadVSURv_1_4.wav', mel_bins=128)
input_tdim = 512
label_csv = '../egs/spotify_dataset/data/class_labels_indices.csv'      
labels = load_label(label_csv)


feats_data = make_features('../sp/7mPXeVKLdc6g7gmadVSURv_1_4.wav', mel_bins=128)
print(feats_data.shape)
batch_size = 2
feats = feats_data.unsqueeze(1)
#print("Dps do unsqueeze:",feats.shape)
features = feats.transpose(1,0)
#print("Dps do transpose:",features.shape)
features = features.to(torch.device("cuda:0"))
print(features.shape)
outputs = audio_model(features)
print(outputs)
# print(feats.shape)
# test_input = torch.rand([2, input_tdim, 128])
# test_output = audio_model.forward(features)
#print(test_output)

with torch.no_grad():
  with autocast():
    output = audio_model.forward(features)
    output = torch.sigmoid(output)
result_output = output.data.cpu().numpy()[0]
sorted_indexes = np.argsort(result_output)[::-1]
print(sorted_indexes)
print('Predice results:')
for k in range(2):
    print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))



