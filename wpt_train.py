from dataloader.dataloader import get_dataLoader, custom_collate_fn_wpt
from model.wptnet import WPTNet
from configuration import config
from model.loss import MSEPathLoss
from dataloader.mapsample import MapSample
from tqdm import tqdm
import torch
import numpy as np
from utils.logger import SummaryLogger

def test(model, test_dataloader, device, epoch=0):
    model.eval()
    
    with tqdm(test_dataloader, desc=f'Test Epoch {epoch}', total=test_dataloader.dataset.__len__()) as tbar:
        for step, (map, start, goal, path, filename, patharray) in enumerate(test_dataloader):
            batch_size = map.shape[0]

            if start.dtype != torch.long:
                start, goal = start.long(), goal.long()
            if device != 'cpu':
                map, start, goal, path = map.to(device), start.to(device), goal.to(device), path.to(device)
            
            out = model(map, start, goal)

            sample_data = []
            for j in range(batch_size):
                map_np = map[j].squeeze().cpu().detach().numpy()
                start_np = start[j].cpu().detach().numpy()
                goal_np = goal[j].cpu().detach().numpy()
                path_np = patharray[j].cpu().detach().numpy()
                sample_data.append(MapSample.get_bgr_map(map_np, start_np, goal_np, path_np))
                tbar.update()

            SummaryLogger.tensorboard.add_images("sample/data", np.stack(sample_data), epoch, dataformats='NHWC')
            SummaryLogger.tensorboard.add_images("sample/prediction", out, epoch, dataformats='NCHW')

def valid(model, val_dataloader,
        loss_function, device,
        scheduler=None, epoch=0):
    
    model.eval()
    loss_per_step = [] 
    with tqdm(total=len(val_dataloader), desc=f'Valid Epoch {epoch}') as tbar:
        for step, (map, goal, path_array) in enumerate(val_dataloader):

            if device != 'cpu':
                map, goal, path_array = map.to(device), goal.to(device), path_array.to(device)
            
            out = model(map, path_array, goal)
            loss = loss_function(out, path_array)
            loss_per_step.append(loss.cpu().item())

            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(loss)

            tbar.set_postfix(loss=loss.cpu().item(), step_num=step)
            tbar.update()

    avg_loss = sum(loss_per_step)/len(loss_per_step)
    SummaryLogger.logger.info(f"[Epoch {epoch}]\t Valid loss: {avg_loss}")

    return loss

def train(model, train_dataloader, loss_function, optimizer, epochs, device, 
        val_dataloader = None, test_dataloadet=None, save_every=1, scheduler=None):
    
    for epoch in range(epochs):
        model.train()
        loss_per_step = []
        with tqdm(total=len(train_dataloader), desc=f'Train Epoch {epoch}') as tbar:
            for step, (map, goal, path_array) in enumerate(train_dataloader):

                if device != 'cpu':
                    map, goal, path_array = map.to(device), goal.to(device), path_array.to(device)
                
                optimizer.zero_grad()
                out = model(map, path_array, goal)
                loss = loss_function(out, path_array)
                loss.backward()
                loss_per_step.append(loss.cpu().item())
                
                optimizer.step()
                if scheduler is not None and type(scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step()
                
                tbar.set_postfix(loss=loss.cpu().item(), step_num=step)
                tbar.update()

        train_loss = sum(loss_per_step)/len(loss_per_step)

        SummaryLogger.logger.info(f"[Epoch {epoch}]\t Train average loss: {train_loss}")
        SummaryLogger.tensorboard.add_scalar("train/loss", train_loss, global_step=epoch)

        if val_dataloader is not None:
            eval_loss = valid(model, val_dataloader, loss_functiion, device, epoch=epoch)
            SummaryLogger.tensorboard.add_scalar("valid/loss", eval_loss, global_step=epoch)
        
        # if test_dataloader is not None:
        #     test(model, test_dataloader, device, epoch=epoch)
        
        if epoch > 0 and epoch % save_every == 0:
            SummaryLogger.save_model(model, f"model_{epoch}.pth")


if __name__ == "__main__":
    SummaryLogger.init()

    model = WPTNet().to(config.device)
    # SummaryLogger.load_model(model, r"E:\code\navigation\trainLogs\2024_09_15_20_03_48\checkpoints\model_9.pth")

    train_dataloder = get_dataLoader("train_debug", collate_fn=custom_collate_fn_wpt)
    valid_dataloader = get_dataLoader("train_debug", collate_fn=custom_collate_fn_wpt)
    test_dataloader = get_dataLoader("test_debug")
    loss_functiion = MSEPathLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train(model, train_dataloder, loss_functiion, optimizer, config.train_epochs, config.device, val_dataloader=valid_dataloader, test_dataloadet=test_dataloader)
    # test(model, test_dataloader, config.device)

    SummaryLogger.close()
    pass