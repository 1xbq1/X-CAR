import os, warnings, logging, pynvml, torch, numpy as np
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from . import model
from . import scheduler
import torch.nn as nn
from torch.nn import LSTM, RNN, GRU
import math
from math import sin,cos,log,pow
import torch.nn.functional as F
import copy
from .dataset.tools import aug_look, ToTensor
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(q, z, N):
    q = F.normalize(q, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)
    # [N, N]
    sim_matrix = torch.mm(q, z.t().contiguous())
    mask = (torch.ones_like(sim_matrix) - torch.eye(N, device=sim_matrix.device)).bool()
    # [N, N-1]
    sim_matrix = sim_matrix.masked_select(mask).view(N, -1)
    return (q * z).sum(dim=-1) - sim_matrix.sum(dim=-1) / (N - 1)

class projection_MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 0
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        x_net = x
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 0:
            return x, x_net
        else:
            raise Exception
        return x, x_net


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        #x = self.layer1(x)
        #x = self.layer2(x)
        return x 


def aug_rotate():

    angle = math.radians(30)
    
    R_x = torch.tensor([[1, 0, 0],
                     [0, cos(angle), sin(angle)],
                     [0, -sin(angle), cos(angle)]])
    R_x = R_x.transpose(0,1)
    
    R_y = torch.tensor([[cos(angle), 0, -sin(angle)],
                     [0, 1, 0],
                     [sin(angle), 0, cos(angle)]])
    R_y = R_y.transpose(0,1)
    
    R_z = torch.tensor([[cos(angle), sin(angle), 0],
                     [-sin(angle), cos(angle), 0],
                     [0, 0, 1]])
    R_z = R_z.transpose(0,1)
    R = torch.matmul(torch.matmul(R_x,R_y),R_z)
    return R.cuda()

def aug_shear():

    sh_next = 1
    R = torch.tensor([[1, sh_next, sh_next],
                     [sh_next, 1, sh_next],
                     [sh_next, sh_next, 1]])
    R = R.transpose(0,1)
    return R.cuda()

def aug_scale():
    sc_next = 2
    R = torch.tensor([[sc_next, 0, 0],
                     [0, sc_next, 0],
                     [0, 0, sc_next]])
    R = R.transpose(0,1)
    return R.cuda()

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.rotate_weight = torch.nn.Parameter(torch.FloatTensor(3,3), requires_grad=True)
        self.rotate_weight.data.fill_(0.0)
        self.rotate_weight.data[0][0].fill_(1.0)
        self.rotate_weight.data[1][1].fill_(0.7)
        self.rotate_weight.data[1][2].fill_(0.7)
        self.rotate_weight.data[2][1].fill_(0.7)
        self.rotate_weight.data[2][2].fill_(0.7)
        self.R_rotate = aug_rotate()
        
        self.shear_weight = torch.nn.Parameter(torch.FloatTensor(3,3), requires_grad=True)
        self.shear_weight.data[0][0].fill_(1.0)
        self.shear_weight.data[1][1].fill_(1.0)
        self.shear_weight.data[2][2].fill_(1.0)
        self.shear_weight.data[0][1].fill_(0.8)
        self.shear_weight.data[0][2].fill_(0.8)
        self.shear_weight.data[1][0].fill_(0.8)
        self.shear_weight.data[1][2].fill_(0.8)
        self.shear_weight.data[2][0].fill_(0.8)
        self.shear_weight.data[2][1].fill_(0.8)
        self.R_shear = aug_shear()
        
        self.scale_weight = torch.nn.Parameter(torch.FloatTensor(3,3), requires_grad=True)
        self.scale_weight.data.fill_(0.0)
        self.scale_weight.data[0][0].fill_(0.6)
        self.scale_weight.data[1][1].fill_(0.6)
        self.scale_weight.data[2][2].fill_(0.6)
        self.R_scale = aug_scale()
        
        self.backbone = backbone
        self.projector = projection_MLP()

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        cc = C//2
        
        x1 = x
        x_mid = x1
        
        for i in range(I):
            for j in range(N):
            
                R_rotate = self.R_rotate * self.rotate_weight
                
                x_mid_1 = x1[j,i,:cc,:,:,:]
                x_mid_1 = torch.matmul(x_mid_1.permute([1, 2, 3, 0]), R_rotate)
                x_mid_1 = x_mid_1.permute(3, 0, 1, 2)
                
                x_mid_2 = x1[j,i,cc:,:,:,:]
                x_mid_2 = torch.matmul(x_mid_2.permute([1, 2, 3, 0]), R_rotate)
                x_mid_2 = x_mid_2.permute(3, 0, 1, 2)
                
                x1[j,i,:,:,:,:] = torch.cat((x_mid_1, x_mid_2), dim=0)


        for i in range(I):
            for j in range(N):
                
                R_shear = self.R_shear * self.shear_weight
            
                x_mid_1 = x1[j,i,:cc,:,:,:]
                x_mid_1 = torch.matmul(x_mid_1.permute([1, 2, 3, 0]), R_shear)
                x_mid_1 = x_mid_1.permute(3, 0, 1, 2)
                
                x_mid_2 = x1[j,i,cc:,:,:,:]
                x_mid_2 = torch.matmul(x_mid_2.permute([1, 2, 3, 0]), R_shear)
                x_mid_2 = x_mid_2.permute(3, 0, 1, 2)
                
                x1[j,i,:,:,:,:] = torch.cat((x_mid_1, x_mid_2), dim=0)


        for i in range(I):
            for j in range(N):
            
                R_scale = self.R_scale * self.scale_weight
            
                x_mid_1 = x1[j,i,:cc,:,:,:]
                x_mid_1 = torch.matmul(x_mid_1.permute([1, 2, 3, 0]), R_scale)
                x_mid_1 = x_mid_1.permute(3, 0, 1, 2)
                
                x_mid_2 = x1[j,i,cc:,:,:,:]
                x_mid_2 = torch.matmul(x_mid_2.permute([1, 2, 3, 0]), R_scale)
                x_mid_2 = x_mid_2.permute(3, 0, 1, 2)
                
                x1[j,i,:,:,:,:] = torch.cat((x_mid_1, x_mid_2), dim=0)
        
        f, h = self.encoder, self.predictor
        x2 = x
        z1, out1 = f(x1)
        z2, out2 = f(x2)
        q1, q2 = h(z1), h(z2)
        
        #L = D(p1, z2) / 2 + D(p2, z1) / 2
        loss_one = loss_fn(q1, z2.detach(), N)
        loss_two = loss_fn(q2, z1.detach(), N)

        loss = loss_one + loss_two
        
        return loss.mean(), out1, out2

class Initializer():
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.model_name = '{}_{}_{}'.format(self.args.config, self.args.model_type, self.args.dataset)
            #self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            for i in self.args.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memused = meminfo.used / 1024 / 1024
                logging.info('GPU-{} used: {}MB'.format(i, memused))
                if memused > 1000:
                    pynvml.nvmlShutdown()
                    logging.info('')
                    logging.error('GPU-{} is occupied!'.format(i))
                    raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.debug, self.args.dataset, **dataset_args
        )
        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=True, drop_last=True
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=False, drop_last=False
        )
        self.location_loader = self.feeders['ntu_location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': [torch.Tensor(part).long() for part in self.parts]
        }
        self.model_sub = model.create(self.args.model_type, **(self.args.model_args), **kwargs).to(self.device)
        self.model = SimSiam(self.model_sub).to(self.device)
        '''self.model = torch.nn.DataParallel(
            self.model_s, device_ids=self.args.gpus, output_device=self.output_device
        )'''
        self.classifier = nn.Linear(256, self.num_class).to(self.device)
        
        #self.classifier = torch.nn.DataParallel(self.classifier)
        #self.model1 = self.model
        '''self.model1 = model.create(self.args.model_type, **(self.args.model_args), **kwargs).to(self.device)
        self.model1 = torch.nn.DataParallel(
            self.model1, device_ids=self.args.gpus, output_device=self.output_device
        )'''
        
        #n_data = len(self.feeders['train'])
        #nce_k = 16384
        '''nce_t = 0.06
        softmax = True
        #self.contrast = MemoryMoCo(60, nce_k, nce_t, softmax).cuda(self.device)
        self.criterion = NCESoftmaxLoss().cuda(self.device)
        self.contrast = End_to_end(nce_t, softmax)'''
        
        #moment_update(self.model, self.model1, 0)
        
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        logging.info('Model parameters: {:.2f}M'.format(
            sum(p.numel() for p in self.model.parameters()) / 1000 / 1000
        ))
        pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'))
            self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}'.format(pretrained_model))

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.info('Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        if self.args.byol:
            self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        else:
            self.optimizer = optimizer([{'params':self.model.parameters()},{'params':self.classifier.parameters()}], **optimizer_args)
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))
