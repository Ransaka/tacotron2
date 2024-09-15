from datetime import datetime
import os
import time
import argparse
import math
from numpy import finfo
from omegaconf import OmegaConf
import wandb
import pandas as pd
from huggingface_hub import upload_file
import torch
import matplotlib.pyplot as plt
from typing import Tuple
import json

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from text import SinhalaTokenizerTacotron
from audio_processing import inverse_mel_spec_to_wav


class ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        return super().default(obj)
    

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    train_metadata_file = os.path.join(hparams.data_dir, hparams.training_files)
    val_metadata_file = os.path.join(hparams.data_dir, hparams.validation_files)
    _train_and_val_files = pd.concat([pd.read_csv(fp) for fp in [train_metadata_file, val_metadata_file]])
    _text_lines = _train_and_val_files[hparams.text_column_name].values.tolist()
    _tokenizer = SinhalaTokenizerTacotron(text_list=_text_lines)
    trainset = TextMelLoader(train_metadata_file, _tokenizer.vocab_map ,hparams)
    valset = TextMelLoader(val_metadata_file, _tokenizer.vocab_map,  hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn, _tokenizer.vocab_map


def load_model(hparams):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Tacotron2(hparams).to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath, to_hf=True):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save(
        {'iteration': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'char_map': model.char_map
        },
        filepath
    )
    if to_hf:
        upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filepath.split("/")[-1],
            repo_type='model',
            repo_id=hparams.hf_repo_id,
            token=hparams.hf_api_key
        )


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    audio, fig = inference_utterance(model)
    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        if logger:
            logger.log({"val_loss": val_loss})
            logger.log({"iteration": iteration})
            logger.log({"Image/utterance": wandb.Image(fig, caption="Example Audio Mel Spec")})
            logger.log({"Audio/audio": wandb.Audio(audio, sample_rate=hparams.sample_rate, caption="Example Audio Generated")})


def inference_utterance(
        model, 
        text: str='ලබන වසරේ', 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> Tuple[torch.Tensor, plt.Figure]:
    model.eval()
    
    # Convert text to sequence
    sequence = torch.tensor(model.text_to_sequence(text, truncate_and_pad=False)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model.inference(sequence)
    
    mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mel_outputs_postnet[0].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
    ax.set_title(f'Mel Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mel Frequency')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    
    audio = inverse_mel_spec_to_wav(mel_outputs_postnet[0].cpu())
    
    return audio, fig


def train(output_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    logging = hparams.logging
    logger = None
    if logging:
        wandb.login(key=hparams.wandb_api_key)
        config_dict = json.loads(json.dumps(hparams, cls=ConfigEncoder))
        logger = wandb.init(
            project="Tacotron2", 
            config=config_dict, 
            name=hparams.run_name+"-"+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    train_loader, valset, collate_fn, vocab_map = prepare_dataloaders(hparams)

    #load learned tkenization map
    model.load_tokenizer(char_map=vocab_map)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                if logging:
                    logger.log({"train_loss": reduced_loss, "grad_norm": grad_norm, "duration": duration, "iteration": iteration})

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path, hparams.push_to_hub)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default='out',
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = OmegaConf.load('config.yaml')

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)