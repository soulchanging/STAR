# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils

# only change critic loss based on drqv2
# change pi(s) to p(s,zs);Q(s,a) to Q(s,a,zs,zsa)
def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class SucEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=128, hdim=1024, activ=F.elu):
        super(SucEncoder, self).__init__()
        self.activ = activ
        self.zs_dim = zs_dim
        self.ln = nn.LayerNorm(zs_dim)
        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

        self.r1 = nn.Linear(zs_dim+action_dim, hdim)
        self.r2 = nn.Linear(hdim, hdim)
        self.r3 = nn.Linear(hdim, 1)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = self.ln(self.zs3(zs))
        return zs


    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa
    
    def r(self,zs,action):
        r = self.activ(self.r1(torch.cat([zs, action], 1)))
        r = self.activ(self.r2(r))
        r = self.r3(r)
        return r

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim,zs_dim=128):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, 256),
                                   nn.LayerNorm(256), nn.Tanh())


        # self.p0 = nn.Linear(feature_dim,256)

        self.trunk2 = nn.Sequential(nn.Linear(zs_dim+256, feature_dim),
                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        
        self.apply(utils.weight_init)

    def forward(self, obs, std,zs):
        h = self.trunk(obs)
        # h = self.p0(h)
        # h =  nn.LayerNorm(self.p0(h))
        i = torch.cat([h,zs],1)
        i = self.trunk2(i)

        mu = self.policy(i)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim,zs_dim=128):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, 256),
                                   nn.LayerNorm(256), nn.Tanh())
        # self.ln = nn.LayerNorm(256)
        self.trunk2 = nn.Sequential(nn.Linear(zs_dim*2+256, feature_dim),
                nn.LayerNorm(feature_dim), nn.Tanh())


        # self.Q01 = nn.Linear(, 256)
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        # self.Q02 = nn.Linear(feature_dim + action_shape[0]+2*zs_dim, 256)
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action,zsa,zs):
        h = self.trunk(obs)
        h= torch.cat([h,zsa,zs],1)
        h = self.trunk2(h)
        h_action = torch.cat([h,action],1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,alpha):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.alpha = alpha

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.sucencoder = SucEncoder(self.encoder.repr_dim,action_shape[0]).to(device)
        self.fixed_sucencoder = copy.deepcopy(self.sucencoder)
        self.fixed_sucencoder_target = copy.deepcopy(self.sucencoder)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.sucencoder_opt = torch.optim.Adam(self.sucencoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.sucencoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        zs = self.fixed_sucencoder.zs(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev,zs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            fixed_target_zs = self.fixed_sucencoder_target.zs(next_obs)
            dist = self.actor(next_obs, stddev,fixed_target_zs)
            next_action = dist.sample(clip=self.stddev_clip)
            
            
            fixed_target_zsa = self.fixed_sucencoder_target.zsa(fixed_target_zs,next_action)
            
            
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,fixed_target_zsa,fixed_target_zs)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

            fixed_zs = self.fixed_sucencoder.zs(obs)
            fixed_zsa = self.fixed_sucencoder.zsa(fixed_zs,action)

        Q1, Q2 = self.critic(obs, action,fixed_zsa,fixed_zs)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        # r = self.sucencoder.r(obs,action)
        # print(r.shape,reward.shape)
        # print(F.mse_loss(r,reward))
        # r_loss =  F.mse_loss(r,reward)

        # loss = critic_loss + r_loss
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            # metrics['r_loss'] = r_loss.item()
            # metrics['sucencoder_loss'] = sucencoder_loss.item()
        # optimize encoder and critic


        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # self.sucencoder_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        # self.sucencoder_opt.step()
    

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        with torch.no_grad():
            zs = self.fixed_sucencoder.zs(obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev,zs)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        zsa = self.fixed_sucencoder.zsa(zs,action)
        Q1, Q2 = self.critic(obs, action,zsa,zs)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_encoder(self,obs,action,next_obs,reward,step):
        metrics = dict()
        with torch.no_grad():
            next_zs = self.sucencoder.zs(next_obs)
        zs = self.sucencoder.zs(obs)
        pred_zs = self.sucencoder.zsa(zs,action)
        # and reward information
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            # fixed_target_zs = self.fixed_sucencoder_target.zs(next_obs)
            dist = self.actor(obs, stddev,zs.detach())
            policy_action = dist.sample(clip=self.stddev_clip)
        
        pred_zs = self.sucencoder.zsa(zs,action)
        policy_zsa = self.sucencoder.zsa(zs.detach(),policy_action)
        r = self.sucencoder.r(zs,action)
        r_loss =  F.mse_loss(r,reward)
        zsa_loss = F.mse_loss(pred_zs,next_zs)+F.mse_loss(pred_zs.detach(),policy_zsa)
        repr_loss = zsa_loss + r_loss
        
        self.sucencoder_opt.zero_grad()
        # self.encoder_opt.zero_grad()
        repr_loss.backward()
        self.sucencoder_opt.step()
        # self.encoder_opt.step()
        if self.use_tb:
            metrics['repr_loss'] = repr_loss.item()
            metrics['r_loss'] = r_loss.item()
            metrics['zsa_loss'] = zsa_loss.item()
        return metrics


        

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs_aug = self.aug(obs.float())
        next_obs_aug = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs_aug)
        with torch.no_grad():
            next_obs = self.encoder(next_obs_aug)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        # udpdate encoder
        metrics.update(
            self.update_encoder(obs.detach(),action,next_obs.detach(),reward,step)
        )

        # obs = self.encoder(obs_aug)
        # with torch.no_grad():
        #     next_obs = self.encoder(next_obs_aug)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        # if step % 250 == 0:
            # self.alpha = self.alpha*0.99
        self.fixed_sucencoder_target.load_state_dict(self.fixed_sucencoder.state_dict())
        self.fixed_sucencoder.load_state_dict(self.sucencoder.state_dict())
        # utils.soft_update_params(self.fixed_sucencoder, self.fixed_sucencoder_target,
        #                          self.critic_target_tau)    
        # utils.soft_update_params(self.sucencoder, self.fixed_sucencoder,
        #                          self.critic_target_tau)
        return metrics