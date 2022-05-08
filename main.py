#@title
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from copy import deepcopy
import time

###########################################---settings---#################################################    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DataSet_Name = 'MNIST' 
BatchSize = 10

epochnum = 3  
ROUND = 10 #10 
lr_init = 0.005 
gamma = 0.5 
random_seed_fixed = 'yes' # 'no'#


K = 50 #user number
data_num_per_user = 500  
data_num_at_server = 2000 
cross_split = 0.4 

ACC_PSO = 0.2 
LOCAL_ACC_PSO = 1 
GLOBAL_ACC_PSO = 2

full_local_global_rand = 1

FedAvg_on_full = 1 # FedAvg switch
FedPSO_on = 1      # FedPSO switch
###########################################---settings---#################################################  


#======================================================================================================
# download mnist or cifar10 as training and testing datasets (60000 or 50000 + 10000)
def load_dataset(DataSet_Name, BatchSize):
    
    if DataSet_Name == 'MNIST':
        transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transf)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transf)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=True, num_workers=0)
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return trainset, trainloader, testset, testloader, classes
#------------------------------------------------------------------------------------------------------


#======================================================================================================
# split train dataset into K+1 pieces (K for agents and 1 for server)
def split_train_dataset_to_agents_server(K, input_trainset, BatchSize, data_num_per_user, data_num_at_server,
                                         cross_split):
    trainset_all_users = [ () for k in range(K+1)]
    trainloader_all_users = [ () for k in range(K+1)]
    trainset_remain = input_trainset
    full_length = len(input_trainset)
    trainset_all_users_train = [ () for k in range(K)]
    trainset_all_users_cross = [ () for k in range(K)]
    trainloader_all_users_train = [ () for k in range(K)]
    trainloader_all_users_cross = [ () for k in range(K)]
    
    for k in range(K):
        remain_len = full_length - data_num_per_user*(k+1)
        trainset_all_users[k], trainset_remain = torch.utils.data.random_split(trainset_remain, 
                                                                               [data_num_per_user, remain_len])
        trainloader_all_users[k] = torch.utils.data.DataLoader(trainset_all_users[k], batch_size=BatchSize, 
                                                               shuffle=True, num_workers=0)
        trainset_all_users_train[k], trainset_all_users_cross[k] = torch.utils.data.random_split(
            trainset_all_users[k], [data_num_per_user-int(cross_split*data_num_per_user), 
                                    int(cross_split*data_num_per_user)])
        trainloader_all_users_train[k] = torch.utils.data.DataLoader(trainset_all_users_train[k], 
                                                                     batch_size=BatchSize, shuffle=True, 
                                                                     num_workers=0)
        trainloader_all_users_cross[k] = torch.utils.data.DataLoader(trainset_all_users_cross[k], 
                                                                     batch_size=BatchSize, shuffle=True, 
                                                                     num_workers=0)
        
    remain_len = remain_len - data_num_at_server
    trainset_all_users[K], trainset_remain = torch.utils.data.random_split(trainset_remain, 
                                                                           [data_num_at_server, remain_len])
    trainloader_all_users[K] = torch.utils.data.DataLoader(trainset_all_users[K], batch_size=BatchSize, 
                                                           shuffle=True, num_workers=0)
    
    
    return trainloader_all_users, trainset_remain, trainloader_all_users_train, trainloader_all_users_cross
#------------------------------------------------------------------------------------------------------


#======================================================================================================
# CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if DataSet_Name == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)                
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        if DataSet_Name == 'MNIST':
            self.fc1 = nn.Linear(16 * 4 * 4, 120)          
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(Func.relu(self.conv1(x)))
        x = self.pool(Func.relu(self.conv2(x)))
        if DataSet_Name == 'MNIST':
            x = x.view(-1, 16 * 4 * 4)                     
        x = Func.relu(self.fc1(x))
        x = Func.relu(self.fc2(x))
        x = self.fc3(x)
        return x   

#======================================================================================================
# SGD 
def basic_client_update(k, agent_net, trainloader, epochnum, lr, gamma):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(agent_net.parameters(), lr=lr, momentum=0.9)  
    print('--------------')
    print('agent-%d of %d: ' % (k, K))
    agent_net = agent_net.to(device)
    for epoch in range(epochnum):
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            if i==len(trainloader):
                print(i)
                break
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = agent_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    lr_new = gamma*lr
    return agent_net, running_loss/(i+1), lr_new
#------------------------------------------------------------------------------------------------------


#======================================================================================================
# average multi-net parameters 
def netavg(users_nets, server_avg_net):
    param_dict = server_avg_net.state_dict()
    param_names = list(param_dict)    
    KK = len(users_nets)
    for name in param_names: # layer-by-layer
        tmp_param_buff = torch.zeros(param_dict[name].shape).to(device)
        for k in range(KK):   # agent-by-agent
            users_nets[k] = users_nets[k].to(device)
            tmp_param_buff += users_nets[k].state_dict()[name]
        server_avg_net.state_dict()[name].copy_(tmp_param_buff/KK)
    return server_avg_net
#------------------------------------------------------------------------------------------------------


#======================================================================================================
# calculate accuracy 
def cal_test_accuracy(dataloader, trained_net):
    correct = 0
    total = 0
    trained_net = trained_net.to(device)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = trained_net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    return test_accuracy       
#------------------------------------------------------------------------------------------------------


#======================================================================================================
def get_best_score_by_FC_loss(server_result):
    temp_score = 100000
    temp_index = 0
    for index, result in enumerate(server_result):
        if temp_score > result[1]:
            temp_score = result[1]
            temp_index = index
    return server_result[temp_index][0], server_result[temp_index][1]
#------------------------------------------------------------------------------------------------------


#======================================================================================================
# PSO 
class particle():
    def __init__(self, client_id, client_net, Agent_trainloader, Agent_cross_trainloader, FC_trainloader, ACC, LOCAL_ACC, GLOBAL_ACC):
        self.client_id = client_id
        self.client_net = client_net.to(device)
        self.local_best_net = client_net.to(device)
        self.global_best_net = client_net.to(device)
        self.local_best_score = 1.0
        self.global_best_score = 1.0
        self.param_dict = client_net.state_dict()
        self.param_names = list(self.param_dict)
        self.velocities = [ [] for i in range(len(self.param_names))]
        for i, name in enumerate(self.param_names):
            shape = client_net.state_dict()[name].shape
            self.velocities[i] = torch.rand(*shape)/5 - 0.1
        self.Agent_trainloader = Agent_trainloader
        self.Agent_cross_trainloader = Agent_cross_trainloader
        self.coeff = {'acc':ACC, 'local_acc':LOCAL_ACC, 'global_acc':GLOBAL_ACC}
        self.FC_trainloader = FC_trainloader
        
    def update_global_net(self, global_best_net, global_best_score):
        if self.global_best_score > global_best_score:
            self.global_best_net = deepcopy(global_best_net).to(device) 
            self.global_best_score = global_best_score
            
    def resp_best_net(self, gid):
        if self.client_id == gid:
            return self.client_net 
        
    def train_particle(self):
        step_model = self.client_net
        step_weight = step_model.state_dict()
        #step_weight = step_weight.to(device)
        new_weight = [None] * len(step_weight)
        #new_weight = new_weight.to(device)
        if full_local_global_rand == 0:
            local_rand, global_rand = torch.rand(1), torch.rand(1)
        for i, layer in enumerate(step_weight):
            new_v = self.coeff['acc'] * self.velocities[i]
            new_v = new_v.to(device)
            if full_local_global_rand == 1:
                shape=self.velocities[i].shape 
                local_rand, global_rand = torch.rand(*shape).to(device), torch.rand(*shape).to(device)
            new_v = new_v + self.coeff['local_acc'] * local_rand * (self.local_best_net.state_dict()[layer] 
                                                                    - step_weight[layer])
            new_v = new_v + self.coeff['global_acc'] * global_rand * (self.global_best_net.state_dict()[layer] 
                                                                      - step_weight[layer])
            self.velocities[i] = new_v
            new_weight[i] = step_weight[layer] + self.velocities[i]
            step_model.state_dict()[layer].copy_(new_weight[i])            
        self.client_net, train_loss_own_data, lr_new = basic_client_update(self.client_id, step_model, 
                                                                           self.Agent_trainloader, epochnum, 
                                                                           lr, gamma)
        Agent_cross_data_accuracy = cal_test_accuracy(self.FC_trainloader, self.client_net)
        train_score_loss_with_agent_cross_data = 1 - Agent_cross_data_accuracy
        if self.local_best_score > train_score_loss_with_agent_cross_data:
            self.local_best_net = self.client_net
            self.local_best_score = train_score_loss_with_agent_cross_data
        FC_global_data_accuracy = cal_test_accuracy(self.FC_trainloader, self.client_net) 
        train_score_loss = 1 - FC_global_data_accuracy
        return self.client_id, train_score_loss, lr_new
#------------------------------------------------------------------------------------------------------




if __name__ == '__main__':
    time_start = time.time()
    
    if random_seed_fixed == 'yes':
        torch.manual_seed(1)
    
    
    trainset, trainloader, testset, testloader, classes = load_dataset(DataSet_Name, BatchSize)
    
    if K*data_num_per_user + data_num_at_server > len(trainset):
        print('!!! ERROR: The trainset is not sufficient to be allocated among users and server !!!')
        sys.exit()
        
    trainloader_all_users, trainset_remain, trainloader_all_users_train, trainloader_all_users_cross = split_train_dataset_to_agents_server(
        K, trainset, BatchSize, data_num_per_user, data_num_at_server, cross_split)
    
    
    print('\n========================== Print Settings ==============================\n')
    print('DataSet =', DataSet_Name)
    print('BatchSize =', BatchSize)
    print('epochnum =', epochnum)
    print('round =', ROUND)
    print('lr_init =', lr_init)
    print('gamma =', gamma)
    print('fixed random seed =', random_seed_fixed)
    print('K =', K)
    print('data # per user =', data_num_per_user)
    print('data # @ server =', data_num_at_server)
    print('unallocated data # =', len(trainset_remain))
    print('cross_split_ratio =', cross_split)
    print('ACC_PSO =', ACC_PSO)
    print('LOCAL_ACC_PSO =', LOCAL_ACC_PSO)
    print('GLOBAL_ACC_PSO =', GLOBAL_ACC_PSO)
    print('\n')
    
    
    net = Net().to(device)
    
    lr = lr_init
    
    
    if FedAvg_on_full == 1:
        fedavg_net_all_users_full = [ [] for k in range(K)]
        for k in range(K):
            fedavg_net_all_users_full[k] = deepcopy(net).to(device)
        FedAvg_server_avg_net_full = deepcopy(net).to(device)
        fedavg_server_evaluate_test_data_full_show = []
    
    
    if FedPSO_on == 1:
        fedpso_net_all_users = [ [] for k in range(K)]
        for k in range(K):
            fedpso_net_all_users[k] = deepcopy(net).to(device)
        pso_model = []
        for k in range(K):
            pso_model.append(particle(client_id=k, client_net=fedpso_net_all_users[k], 
                                      Agent_trainloader=trainloader_all_users_train[k], 
                                      Agent_cross_trainloader=trainloader_all_users_cross[k],
                                      FC_trainloader=trainloader_all_users_cross[k], 
                                      ACC=ACC_PSO, LOCAL_ACC=LOCAL_ACC_PSO, GLOBAL_ACC=GLOBAL_ACC_PSO))
        global_best_net = None
        global_best_score = 1.0
        pso_server_evaluate_test_data = []
        pso_server_evaluate_test_data_check = []
    

    for r in range(ROUND):
        print('\n\n================================================================\n')
        print('                 In round-%d of %d' % (r+1, ROUND))
        print('\n================================================================')
        
        if FedAvg_on_full == 1:    
            print('\n Fed-Avg')
        for k in range(K):
            if FedAvg_on_full == 1:    
                fedavg_net_all_users_full[k], _, lr_new = basic_client_update(
                    k, fedavg_net_all_users_full[k], trainloader_all_users_train[k], epochnum, lr, gamma)
          #FC_global_data_accuracy_fedavg = cal_test_accuracy(trainloader_all_users[K], 
           #                                                        fedavg_net_all_users_full[k]) 
        if FedAvg_on_full == 1:
            FedAvg_server_avg_net_full = netavg(fedavg_net_all_users_full, FedAvg_server_avg_net_full)
            fedavg_server_evaluate_test_data_full_show.append(cal_test_accuracy(testloader,
                                                                                FedAvg_server_avg_net_full))
          
            for k in range(K):
                fedavg_net_all_users_full[k] = deepcopy(FedAvg_server_avg_net_full).to(device)
        
        
        if FedPSO_on == 1:
            print('\n Fed-PSO')
            FedPSO_server_result = []
            for client in pso_model:
                if r != 0:
                    client.update_global_net(global_best_net, global_best_score) ###client.update_global_net(FedPSO_server_best_net, global_best_score)
                pid, train_score_loss, lr_new = client.train_particle()
                FedPSO_server_result.append([pid, train_score_loss])
            gid, global_best_score = get_best_score_by_FC_loss(FedPSO_server_result)    
            for client in pso_model:
                if client.resp_best_net(gid) != None:
                    global_best_net = client.resp_best_net(gid)
            pso_server_evaluate_test_data.append(cal_test_accuracy(testloader, global_best_net)) ### pso_server_evaluate_test_data.append(cal_test_accuracy(testloader, FedPSO_server_best_net))
            
        lr = lr_new
    
    
    time_end = time.time()
    print('\n~~~~~~Finish~~~~~~\n')
    print('time cost', time_end-time_start, 'seconds')
    
    
    if FedAvg_on_full == 1:
        print('fedavg_test_accuracy = \n', fedavg_server_evaluate_test_data_full_show)
        x_axle = torch.zeros(1,ROUND)
        for rr in range(ROUND):
            x_axle[0,rr] = rr +1

        plt.plot(x_axle[0,:], fedavg_server_evaluate_test_data_full_show, label='FedAvg')
        plt.legend()
        plt.grid(True)
    
    if FedPSO_on == 1:
        print('pso_test_accuracy = \n', pso_server_evaluate_test_data)
        x_axle = torch.zeros(1,ROUND)
        for rr in range(ROUND):
            x_axle[0,rr] = rr +1

        plt.plot(x_axle[0,:], pso_server_evaluate_test_data, label='FedPSO')
        plt.legend()
        plt.grid(True)
