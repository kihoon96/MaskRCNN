from dataloader import Dataloader_COCO
from torch.utils.data import DataLoader

def main():
    cocoloader = Dataloader_COCO('test')
    self.batch_generator = DataLoader(dataset=cocoloader, batch_size=8, shuffle = True, num_workers = 8)
    
