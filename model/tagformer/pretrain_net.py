import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ruamel_yaml as yaml
from accelerate import Accelerator
import torch
from torch.optim import AdamW
from transformers.optimization import (
    get_polynomial_decay_schedule_with_warmup,
)

from dataset_proc.load_dataset import load_train_valid_dataset_stage_align

from models.model_pretrain import RTL_Fusion
from models.model_net import Net_Encoder

from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter  


date ='pretrain_net_1020'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

def main(args, config):

    max_epoch = config['schedular']['epochs']
    
    valid_net_loader = load_train_valid_dataset_stage_align(batch_size=config['batch_size'], idx=None, train_valid="demo")
    train_net_loader = load_train_valid_dataset_stage_align(batch_size=config['batch_size'], idx=None, train_valid="train")
    #### Model ####
    # rtl_fusion = RTL_Fusion(config=config, device=accelerator.device, accelerator=accelerator)
    net_enc = Net_Encoder(config=config, device=accelerator.device, accelerator=accelerator)
    optimizer = AdamW([
        {'params': net_enc.parameters(), 'lr': config['optimizer']['lr'], 'eps':config['optimizer']['eps'], 'weight_decay':config['optimizer']['weight_decay']}
    ])
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['schedular']['warmup_updates'],
        num_training_steps=config['schedular']['total_updates']*max_epoch,
        lr_end=config['schedular']['lr_end'],
        power=config['schedular']['power'],
    )


    (
        net_enc,
        optimizer,
        valid_net_loader,
        train_net_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        net_enc,
        optimizer,  
        valid_net_loader,
        train_net_loader,
        lr_scheduler,
    )

    step = 0

    for epoch in range(max_epoch):
        epoch_loss_train, epoch_loss_valid = 0, 0
        epoch_loss_cl_net, epoch_loss_gmae, epoch_loss_node_cnt = 0, 0, 0
        epoch_loss_cl_rtl, epoch_loss_cl_layout = 0, 0
        real_batch_loss = 0

        j = 0
        ### train ###
        rtl_loader_train, net_ori_loader_train, graph_pos_loader_train, layout_loader_train = train_net_loader

        for idx, data in enumerate(zip(rtl_loader_train, net_ori_loader_train, graph_pos_loader_train, layout_loader_train)):
            step += 1
            # accelerator.print(f"Epoch {epoch}, Step {idx}")
            # try:
            net_enc.train()
            loss_train, loss_cl_net, loss_gmae, loss_node_cnt, loss_cl_rtl, loss_cl_layout = net_enc(data, mode='pretrain')
            accelerator.backward(loss_train)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            # accelerator.print(f"    Train Loss: {loss_train.item()}")
            epoch_loss_train += loss_train.item()
            epoch_loss_cl_net += loss_cl_net.item()
            epoch_loss_gmae += loss_gmae.item()
            epoch_loss_node_cnt += loss_node_cnt.item()
            epoch_loss_cl_rtl += loss_cl_rtl.item()
            epoch_loss_cl_layout += loss_cl_layout.item()
            real_batch_loss += loss_train.item()
            if (idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                # val_loss = valid_network(encoder_rtl, text_enc, text_proj, val_rtl_loader, loss_fn_valid, lambda_cs, lambda_mae)
                accelerator.print(
                    f"Batch{j} train_loss={(real_batch_loss/accumulation_steps):.3f}"
                )
                j+=1
                real_batch_loss = 0
            
        ### valid ###
        with torch.no_grad():
            net_enc.eval()
            rtl_loader_val, net_ori_loader_val, graph_pos_loader_val, layout_loader_val = valid_net_loader
            for idx_val, data in enumerate(zip(rtl_loader_val, net_ori_loader_val, graph_pos_loader_val, layout_loader_val)):
                loss_val,_,_,_,_,_ = net_enc(data, mode='pretrain')
                epoch_loss_valid += loss_val.item()

        epoch_loss_train = epoch_loss_train/(idx+1)
        epoch_loss_valid = epoch_loss_valid/(idx_val+1)
        accelerator.print(f"Epoch {epoch + 1}/{max_epoch}, Total Train Loss: {epoch_loss_train}, Total Val Loss: {epoch_loss_valid}")
        
        writer.add_scalar('Epoch Train Loss', epoch_loss_train, epoch)
        writer.add_scalar('Epoch Valid Loss', epoch_loss_valid, epoch)
        writer.add_scalar('Epoch Loss CL Net', epoch_loss_cl_net, epoch)
        writer.add_scalar('Epoch Loss GMAE', epoch_loss_gmae, epoch)
        writer.add_scalar('Epoch Loss Node Count', epoch_loss_node_cnt, epoch)
        writer.add_scalar('Epoch Loss CL RTL', epoch_loss_cl_rtl, epoch)
        writer.add_scalar('Epoch Loss CL Layout', epoch_loss_cl_layout, epoch)

        ## save model every k epoch###
        k = 3
        if (epoch+1) % k == 0:
            accelerator.wait_for_everyone()
            unwrap_rtl_fusion = accelerator.unwrap_model(net_enc)
            torch.save(unwrap_rtl_fusion.state_dict(), f"{model_save_dir}/net_enc.{epoch}.pt")


if __name__ == '__main__':
    log_dir = f'./log/log_{date}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        os.system(f'rm -r {log_dir}')
        os.mkdir(log_dir)

    model_save_dir = f"./pretrain_model/{date}"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    global writer
    writer = SummaryWriter(log_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/PretrainStage.yaml')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    global accumulation_steps
    accumulation_steps = 32

    main(args, config)