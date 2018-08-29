from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


dev = None

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

#, device=dev
from plot import draw22


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0, dra = False):
        outputs = []
        h_t = torch.zeros(input.size(0), 51).to(device=dev)
        c_t = torch.zeros(input.size(0), 51).to(device=dev)
        h_t2 = torch.zeros(input.size(0), 51).to(device=dev)
        c_t2 = torch.zeros(input.size(0), 51).to(device=dev)
        
#        init_method = nn.init.normal_
##        init_method = nn.init.eye_
#        
#        init_method(self.lstm1.weight_hh)
#        init_method(self.lstm1.weight_ih)
#        
#        init_method(self.lstm2.weight_hh)
#        init_method(self.lstm2.weight_ih)
        
        
        if dra :
            draw22([ h_t.detach().cpu().numpy() ,c_t.detach().cpu().numpy() ,h_t2.detach().cpu().numpy() ,c_t2.detach().cpu().numpy()  ])

        
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            
            if dra and i%10 == 0:
                draw22([ h_t.detach().cpu().numpy() ,c_t.detach().cpu().numpy() ,h_t2.detach().cpu().numpy() ,c_t2.detach().cpu().numpy()  ])
#            print('a')
            
            
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            
            if dra and i%10 == 0:
                draw22([ h_t.detach().cpu().numpy() ,c_t.detach().cpu().numpy() ,h_t2.detach().cpu().numpy() ,c_t2.detach().cpu().numpy()  ])
#            print('a')
            
            
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata.pt')
    input = torch.from_numpy(data[3:, :-1]).to(device=dev)
    target = torch.from_numpy(data[3:, 1:]).to(device=dev)
    test_input = torch.from_numpy(data[:3, :-1]).to(device=dev)
    test_target = torch.from_numpy(data[:3, 1:]).to(device=dev)
    # build the model
    seq = Sequence().to(device=dev)
#    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.Adam(seq.parameters(), lr=0.01)
    #begin to train
    for i in range(100):
        print('STEP: ', i)
        if i == 90 or  i == 50:
            dra=True
        else:
            dra =False
            
            
        def closure():
            optimizer.zero_grad()
            out = seq(input,dra=dra)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future,dra=dra)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().cpu().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
