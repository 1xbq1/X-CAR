import logging, torch, numpy as np
from tqdm import tqdm
from time import time

from . import utils as U
from .initializer import Initializer

from .dataset.tools import aug_look, ToTensor
from torchvision import transforms
import torch.nn as nn
import math
import torch.nn.functional as F 

def aug_transfrom(aug_name, args_list, selected_frames):
    aug_name_list = aug_name.split("_")
    transform_aug = [aug_look('selectFrames', selected_frames)]
    if aug_name_list[0] != 'None':
        for i, aug in enumerate(aug_name_list):
            transform_aug.append(aug_look(aug, args_list[i * 2], args_list[i * 2 + 1]))
    transform_aug.extend([ToTensor(), ])
    transform_aug = transforms.Compose(transform_aug)
    return transform_aug

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)
        
def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds

class Processor(Initializer):

    def train(self, epoch):
        self.model.train()
        self.classifier.train()
        #self.model1.train()
        start_train_time = time()
        num_top1, num_sample = 0, 0
        
        #增加对比学习中的数据变换
        '''args1 = [1, None, None, None, None, None, None, None, None, None, ]
        aug1 = 'subtract_randomFlip_shear' '''
        args1 = [None, None, None, None, None, None, None, None, ]
        aug1 = 'rotate'
        selected_frames = 300
        transform1 = aug_transfrom(aug1, args1, selected_frames)
        args2 = args1
        aug2 = aug1
        transform2 = aug_transfrom(aug2, args2, selected_frames)
        
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)
        for num, (x, y, _) in enumerate(train_iter):
            self.optimizer.zero_grad()

            #N, I, C, T, V, M = x.size()
            #cc = C//2
 
            # Using GPU
            x = x.float().to(self.device)
            
            y = y.long().to(self.device)

            #for name, param in self.model.named_parameters():
                #logging.info('name:{}, param:{}'.format(name, param))
            L, out1, out2= self.model(x)
            #_, out = self.model.encoder(x)
                #out1, out2 = torch.split(out, [N, N], dim=0)
                #f1, f2 = torch.split(z1, [N, N], dim=0)
                #features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            out1 = self.classifier(out1)
            out2 = self.classifier(out2)
            out = out1 + out2
            loss_recog = self.loss_func(out, y)

            #shuffle_ids, reverse_ids = get_shuffle_ids(N)

            # Calculating Output
            #out1, _ = self.model(x1)
            
            #x2 = x2[shuffle_ids]
            #out2, _ = self.model1(x2)
            #out2 = out2[reverse_ids]
                
            # Updating Weights
            #loss_class_1 = self.loss_func(out1, y)
            #loss_class_2 = self.loss_func(out2, y)
            #loss_ER = torch.abs(loss_class_1 - loss_class_2)
            
            #out12 = self.contrast(out1, out2)
            #loss_contrast = self.criterion(out12)
            
            '''out = out1+out2
            loss_recog = self.loss_func(out, y)'''
            
            #loss_recog = (loss_class_1 + loss_class_2) / 2
            #loss = loss_recog + loss_ER
            #loss = loss_recog + L
            loss = L + loss_recog
            #loss = loss_contrast
            loss.backward()
            #print('leaf',self.model.angle_weight.is_leaf)
            #print('requires_grad',self.model.angle_weight.requires_grad)
            #print('grad',self.model.angle_weight.grad)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            #self.model_s.update_moving_average()
            #moment_update(self.model, self.model1, 0.999)

            # Calculating Recognition Accuracies
            num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            num_top1 += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            '''if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)'''
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, L: {:.4f}, loss_recog: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), L.item(), loss_recog.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        train_acc = num_top1 / num_sample
        #if self.scalar_writer:
        #    self.scalar_writer.add_scalar('train_acc', train_acc, self.global_step)
        logging.info('Epoch: {}/{}, Training accuracy: {:d}/{:d}({:.2%}), Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, num_top1, num_sample, train_acc, time()-start_train_time
        ))
        logging.info('')

    def eval(self):
        self.model.eval()
        self.classifier.eval()
        #self.model1.eval()
        start_eval_time = time()
        with torch.no_grad():
            num_top1, num_top5 = 0, 0
            num_sample, eval_loss = 0, []
            cm = np.zeros((self.num_class, self.num_class))
            
            #增加对比学习中的数据变换
            '''args1 = [1, None, None, None, None, None, None, None, None, None, ]
            aug1 = 'subtract_randomFlip_shear' '''
            '''args1 = [1, None, None, None]
            aug1 = 'subtract1'
            selected_frames = 300
            transform1 = aug_transfrom(aug1, args1, selected_frames)
            args2 = args1
            aug2 = aug1
            transform2 = aug_transfrom(aug2, args2, selected_frames)'''
            
            '''n_data = len(self.feeders['eval'])
            nce_k = 16384
            nce_t = 0.06
            softmax = True
            contrast = MemoryMoCo(60, n_data, nce_k, nce_t, softmax).cuda(self.device)
            criterion = NCESoftmaxLoss()
            criterion = criterion.cuda(self.device)'''
            
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, (x, y, _) in enumerate(eval_iter):

                N, I, C, T, V, M = x.size()
                cc = C//2
                #x1 = x
                '''for i in range(I):
                    for j in range(N):
                        x_mid = []
                        x_mid.append(torch.DoubleTensor(transform1(x[j,i,:cc,:,:,:].numpy())))
                        x_mid.append(torch.DoubleTensor(transform1(x[j,i,cc:,:,:,:].numpy())))
                        x1[j,i,:,:,:,:] = torch.cat(x_mid, dim=0)'''
                #x2 = x
                '''for i in range(I):
                    for j in range(N):
                        x_mid = []
                        x_mid.append(torch.DoubleTensor(transform2(x[j,i,:cc,:,:,:].numpy())))
                        x_mid.append(torch.DoubleTensor(transform2(x[j,i,cc:,:,:,:].numpy())))
                        x2[j,i,:,:,:,:] = torch.cat(x_mid, dim=0)'''
                # Using GPU
                x = x.float().to(self.device)
                #x1 = x1.float().to(self.device)
                #x2 = x2.float().to(self.device)
                y = y.long().to(self.device)


                # Calculating Output
                '''out1, _ = self.model(x1)
                out2, _ = self.model1(x2)'''
                _, out1, out2= self.model(x)
                out1 = self.classifier(out1)
                out2 = self.classifier(out2)
                out = out1 + out2
                    
                #out12 = self.contrast(out1, out2)
                #loss_contrast = self.criterion(out12)

                # Getting Loss
                #loss_class_1 = self.loss_func(out1, y)
                #loss_class_2 = self.loss_func(out2, y)
                #out = out1+out2
                loss = self.loss_func(out, y)
                #loss_ER = torch.abs(loss_class_1 - loss_class_2)
                #loss_recog = (loss_class_1 + loss_class_2) / 2
                #loss = loss_recog + loss_ER
                #loss = loss_recog + loss_contrast
                #loss = loss_recog
                eval_loss.append(loss.item())

                # Calculating Recognition Accuracies
                num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                num_top1 += reco_top1.eq(y).sum().item()
                reco_top5 = torch.topk(out,5)[1]
                num_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                # Calculating Confusion Matrix
                for i in range(x.size(0)):
                    cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        acc_top1 = num_top1 / num_sample
        acc_top5 = num_top5 / num_sample
        eval_loss = sum(eval_loss) / len(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)
        logging.info('Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.format(
            num_top1, num_sample, acc_top1, num_top5, num_sample, acc_top5, eval_loss
        ))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        '''if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)'''

        torch.cuda.empty_cache()
        return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                self.classifier.load_state_dict(checkpoint['classifier'])
                #self.model1.module.load_state_dict(checkpoint['model1'])
                #self.contrast.load_state_dict(checkpoint['contrast'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.load_state_dict(checkpoint['model'])
                self.classifier.load_state_dict(checkpoint['classifier'])
                #self.model1.module.load_state_dict(checkpoint['model1'])
                #self.contrast.load_state_dict(checkpoint['contrast'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    acc_top1, acc_top5, cm = self.eval()
                    if acc_top1 > best_state['acc_top1']:
                        is_best = True
                        best_state.update({'acc_top1':acc_top1, 'acc_top5':acc_top5, 'cm':cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                U.save_checkpoint(
                    self.model.state_dict(),self.classifier.state_dict(),self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name
                )
                logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                    best_state['acc_top1'], U.get_time(time()-start_time)
                ))
                logging.info('')
            logging.info('Finish training!')
            logging.info('')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        cm = checkpoint['best_state']['cm']
        self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Loading Data
        x, y, names = iter(self.eval_loader).next()
        location = self.location_loader.load(names) if self.location_loader else []

        # Calculating Output
        self.model.eval()
        out, feature = self.model(x.float().to(self.device))

        # Processing Data
        data, label = x.numpy(), y.numpy()
        out = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
        weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()

        # Saving Data
        if not self.args.debug:
            U.create_folder('./visualization')
            np.savez('./visualization/extraction_{}.npz'.format(self.args.config),
                data=data, label=label, name=names, out=out, cm=cm,
                feature=feature, weight=weight, location=location
            )
        logging.info('Finish extracting!')
        logging.info('')
