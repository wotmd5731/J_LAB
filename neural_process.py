import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

z_dim = 10
x_dim = 2
y_dim = 1
batch_size = 8

inference_core = nn.Linear(x_dim+y_dim,2*z_dim)
generator_core = nn.Linear(x_dim+z_dim,y_dim)

optimizer = torch.optim.Adam([
    {'params':inference_core.parameters()},
    {'params':generator_core.parameters()}
    ],lr=0.01)

#for episode in range(1):
x=torch.rand([batch_size,x_dim])
y=[]
temp=x.sum(dim=1)
for tt in temp:
    if tt.item()>1:
        y.append([1])
    else:
        y.append([0])
y=torch.Tensor(y)

#prior factor
p_mu, p_std = torch.zeros([z_dim]),torch.ones([z_dim])
prior_distribution = Normal(p_mu,F.softplus(p_std))

#inference state 
z_info = F.sigmoid(inference_core(torch.cat([x,y],dim=1)))

#posterior factor 
q_mu,q_std = torch.split(z_info,z_dim,dim=1)
posterior_distribution = Normal(q_mu,F.softplus(q_std))
#posterior sample
z = posterior_distribution.rsample()

y_pred = F.sigmoid(generator_core(torch.cat([x,z],dim=1)))
print('answer \n',y)
print('pred \n',(y_pred>torch.Tensor([0.5])).float())

D_kl_loss = kl_divergence(posterior_distribution,prior_distribution).sum(1).mean()
print('D_kl : ',D_kl_loss)

reconstruction_loss = F.binary_cross_entropy(y_pred,y)
elbo = reconstruction_loss + D_kl_loss
elbo.backward()
optimizer.step()
optimizer.zero_grad()


