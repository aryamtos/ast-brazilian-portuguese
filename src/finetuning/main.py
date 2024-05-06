import os,csv,argparse,wget,sys,time
import numpy as np
import torch
from finetuning.training import train

import dataloader
from models.ast_models import ASTModel
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))



if __name__ =="__main__":


    input_tdim = 512
    class_num = 2
    fstride = 10
    tstride = 10
    audio_train = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 48, 'timem': 192, 'mixup': 0.5, 'dataset': 'speechcommands', 'mode':'train', 'mean':4.6424894, 'std':4.2628665,
                  'noise': True}


    audio_dev = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands', 'mode':'evaluation', 'mean':4.6424894, 'std':4.2628665,
                    'noise': False}
    
    class_labels_indices="../egs/spotify_dataset/data/class_labels_indices.csv"

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset('../egs/metadados_train_spotify_labels_binary.json', label_csv=class_labels_indices,
                                    audio_conf=audio_train), batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset('../egs/metadados_dev_spotify_labels_binary.json', label_csv=class_labels_indices,
                                    audio_conf=audio_dev), batch_size=2, shuffle=False, num_workers=2, pin_memory=True)


    model_path = "../egs/exp/est-speechcommands-f10-t10-pTrue-b128-lr5e-5/models/best_audio_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(model_path,map_location=device)
    ast_mdl = ASTModel(label_dim=class_num,fstride=10,tstride=10,input_fdim=128,input_tdim=input_tdim,audioset_pretrain=False,model_size='small224',verbose=True)
    print(f'[*INFO] load checkpoint: {sd}')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(sd,strict=True)
    train(audio_model,train_loader,val_loader)