import torch
import torch.nn as nn
import numpy as np
import time

#%% BAOAB
class BAOAB(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, lr, weight_decay, gamma, temperature, epochs, device, meas_freq, start_meas=0, lr_schedule=None):
        super(BAOAB, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.train_loader_loss = torch.utils.data.DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset), num_workers=5, pin_memory=True)
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.T = temperature
        self.epochs = epochs
        self.device = device
        self.meas_freq = meas_freq
        self.start_meas = start_meas
        self.lr_schedule = lr_schedule
        
        self.a = np.exp(-self.gamma*self.lr)  # constants used in integrator
        self.sqrt_aT = np.sqrt( (1 - self.a**2)*self.T )
        
        self.running_loss = None
        self.running_loss_list = []
        self.total_loss = []
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set. 
        Also returns arrays of kinetic energies and squared-L2 norm of network parameters (each column
        corresponds to single network layer).
        """        
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        start_time = time.time()
        print("Starting BAOAB sampling...")     
        
        (data, target) = next(iter(self.train_loader))          # compute initial gradients
        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
        self.fill_gradients(data, target, squeeze, datasize)
        
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.randn(p.size()).to(self.device, non_blocking=True)
        
        accu_train = []
        accu_test = []
        
        for epoch in range(1, self.epochs+1):                   # sampling loop
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                self.update_params_BAOA()    # BAOA-steps
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)   # compute new gradients
                self.fill_gradients(data, target, squeeze, datasize)
                                
                self.update_params_B()     # B-step
        

            print("BAOAB EPOCH {} DONE!".format(epoch))
            if epoch % self.meas_freq == 0 and epoch > self.start_meas:
                self.get_total_loss()
                accu_train += [evaluate(self.model, self.train_loader_loss, self.device, self.criterion)]
                accu_test += [evaluate(self.model, self.test_loader, self.device, self.criterion)]
            
      
            if self.lr_schedule != None and epoch % self.lr_schedule[0]==0:
                self.lr *= self.lr_schedule[1]
        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
        
        return (self.total_loss, accu_train, accu_test)


    
    def take_measurement(self,):
        
        self.running_loss_list += [self.running_loss.detach()]
        

        
    def get_total_loss(self,):
        
        self.model.eval()  # Switch to evaluation mode (no dropout, batchnorm freezing)
        with torch.no_grad():
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader_loss):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                loss += self.criterion(output, target).item()
        self.total_loss += [loss]
        self.model.train()


    
    def update_params_BAOA(self):
        """
        Performs B-, A-, O-, and A-step on model. Forces are assumed to be stored in parameter gradients.
        """      
       
        for p in self.model.parameters():
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step

            p.data.add_(p.buf, alpha=0.5*self.lr)                                   # A-step
            
            eps = torch.randn(p.size()).to(self.device, non_blocking=True)          # O-step
            p.buf.mul_(self.a)                        
            p.buf.add_(eps, alpha=self.sqrt_aT) 
            
            p.data.add_(p.buf, alpha=0.5*self.lr)                                   # A-step                               




    def update_params_B(self):
        """
        Performs B-step on model. Forces are assumed to be stored in parameter gradients.
        """

        for p in self.model.parameters():
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step                                
  


    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        self.running_loss = self.criterion(output, target)*datasize
        self.running_loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay)  




#%% ZBAOABZ
class ZBAOABZ(nn.Module):
    
    def __init__(self, model, train_loader, test_loader, criterion, dtau, weight_decay, gamma, alpha, alpha2, m, M, temperature, epochs, device, meas_freq, start_meas=0, lr_schedule=None):
        super(ZBAOABZ, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.train_loader_loss = torch.utils.data.DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset), num_workers=5, pin_memory=True)
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = None
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.alpha = alpha
        self.alpha2 = alpha2
        self.T = temperature
        self.epochs = epochs
        self.device = device
        self.meas_freq = meas_freq
        self.start_meas = start_meas
        
        self.dtau = dtau
        self.zeta = None
        self.a_gamma = np.exp(-self.gamma)  # constants used in integrator
        self.a = None
        self.sqrt_aT = None
        self.gradnorm = None
        self.alpha_inv = 1/self.alpha
        self.exptau_half = np.exp(-0.5*self.alpha*self.dtau)
        self.r=0.25
        self.m=m
        self.M=M
        self.lr_schedule = lr_schedule
        
        self.running_loss = None
        self.running_loss_list = []
        self.total_loss = []
        self.dt_raw = []
        self.zeta_raw = []
        self.gradnorms = []
        
        
    def train(self):
        """
        Main train/sampling routine. Returns arrays of losses and accuracies on both train and test set. 
        Also returns arrays of kinetic energies and squared-L2 norm of network parameters (each column
        corresponds to single network layer).
        """        
        
        datasize = len(self.train_loader.dataset)
        
        squeeze = True if type(self.criterion) == torch.nn.modules.loss.BCELoss else False   # squeeze network output
                                                                                             # for BCELoss (not required
                                                                                             # for NLLLoss)
        
        if self.criterion.reduction != "mean":
            raise ValueError("Criterion reduction mode must be 'mean'.\n")
        
        
        start_time = time.time()
        print("Starting ZBAOABZ sampling...")     
        
        (data, target) = next(iter(self.train_loader))          # compute initial gradients
        data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
        self.fill_gradients(data, target, squeeze, datasize)
        self.set_gradnorm()
        self.zeta = self.gradnorm
        self.Sundman_transform()
        
        for p in self.model.parameters():                       # create momentum buffers
            p.buf = torch.randn(p.size()).to(self.device, non_blocking=True)
        
        accu_train = []
        accu_test = []
        
        measurement_ctr = 0
        for epoch in range(1, self.epochs+1):                   # sampling loop
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                if measurement_ctr % self.meas_freq == 0:
                    self.take_measurement()                  
                measurement_ctr += 1
                
                self.Z_step()
                self.Sundman_transform()
                self.a = self.a_gamma**(self.lr)
                self.sqrt_aT = torch.sqrt( (1 - self.a**2)*self.T )
                
                self.update_params_BAOA()    # BAOA-steps
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)   # compute new gradients
                self.fill_gradients(data, target, squeeze, datasize)
                
                self.update_params_B()     # B-step
                
                self.set_gradnorm()
                self.Z_step()        
                
                self.Sundman_transform()
           
            if self.lr_schedule != None and epoch % self.lr_schedule[0]==0:
                self.dtau *= self.lr_schedule[1]



            print("ZBAOABZ EPOCH {} DONE!".format(epoch))
            if epoch % self.meas_freq == 0 and epoch > self.start_meas:
                self.get_total_loss()
                accu_train += [evaluate(self.model, self.train_loader_loss, self.device, self.criterion)]
                accu_test += [evaluate(self.model, self.test_loader, self.device, self.criterion)]

        
        end_time = time.time()
        print("Training took {} seconds, i.e {} minutes, with {} seconds per epoch!"
              .format(end_time-start_time, (end_time-start_time)/60, (end_time-start_time)/self.epochs))
                      
        
        self.running_loss_list = np.array([i.cpu() for i in self.running_loss_list])
        self.dt_raw = np.array([i.cpu() for i in self.dt_raw])
        self.zeta_raw = np.array([i.cpu() for i in self.zeta_raw])
        

        return (self.total_loss, accu_train, accu_test, self.dt_raw, self.zeta_raw)

    
    
    def take_measurement(self,):
        self.running_loss_list += [self.running_loss.detach()]
        self.dt_raw += [self.lr]
        self.zeta_raw += [self.zeta]
    
    
    
    def get_total_loss(self,):
        self.model.eval()  # switch to evaluation mode (no dropout, batchnorm freezing)
        with torch.no_grad():
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader_loss):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                output = self.model(data)
                loss += self.criterion(output, target).item()
        self.total_loss += [loss]
        self.model.train()
    
    
    def update_params_BAOA(self):
        """
        Performs B-, A-, O-, and A-step on model. Forces are assumed to be stored in parameter gradients.
        """      
       
        for p in self.model.parameters():
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step

            p.data.add_(p.buf, alpha=0.5*self.lr)                                   # A-step
            
            eps = torch.randn(p.size()).to(self.device, non_blocking=True)          # O-step
            p.buf.mul_(self.a)                        
            p.buf.add_(eps, alpha=self.sqrt_aT) 
            
            p.data.add_(p.buf, alpha=0.5*self.lr)                                   # A-step                               




    def update_params_B(self):
        """
        Performs B-step on model. Forces are assumed to be stored in parameter gradients.
        """

        for p in self.model.parameters():
            p.buf.add_(p.grad.data, alpha=-0.5*self.lr)                             # B-step                                
  


    def fill_gradients(self, data, target, squeeze, datasize):
        """
        Fills gradients of the model on batch (data, target).
        """
        
        self.model.zero_grad()
        output = self.model(data)
        if squeeze: output=output.squeeze()
        self.running_loss = self.criterion(output, target)*datasize
        self.running_loss.backward()
        
        for p in list(self.model.parameters()):                 # weight decay / prior component
            p.grad.data.add_(p.data, alpha=self.weight_decay)  



    def set_gradnorm(self):
        
        self.gradnorm = 0
        for p in list(self.model.parameters()):
            self.gradnorm += torch.sum(p.grad.data**2)
        
        self.gradnorm *= self.alpha2   
    
    def Z_step(self):
        self.zeta = self.exptau_half * self.zeta + self.alpha_inv * (1-self.exptau_half) * self.gradnorm


    
    def Sundman_transform(self):
        zeta_r  =self.zeta**self.r
        self.lr = self.dtau * self.m * (zeta_r + self.M) / (zeta_r + self.m) 




#%% Evaluate
def evaluate(model, data_loader, device, criterion):
    model.eval()

    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            correct += (target == torch.argmax(output, dim=1)).sum().item()  # sum of correct predictions

    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy

