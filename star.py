
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils

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
        self.out_dim = 32 * 35 * 35
        self.repr_dim = 128
        

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        h = self.ln(h)
        return h

class SucEncoder(nn.Module):
    def __init__(self, repr_dim, action_dim, hdim=1024, activ=F.elu):
        super(SucEncoder, self).__init__()
        self.activ = activ
        self.repr_dim = repr_dim
        self.ln = nn.LayerNorm(repr_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(repr_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, repr_dim)

        self.r1 = nn.Linear(repr_dim+ action_dim , hdim)
        self.r2 = nn.Linear(hdim, hdim)
        self.r3 = nn.Linear(hdim, 1)

        self.W = nn.Parameter(torch.rand(repr_dim, repr_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    

    def zsa(self, obs, action):
        zsa = self.activ(self.zsa1(torch.cat([obs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa
    
    def r(self,obs,action):
        r = self.activ(self.r1(torch.cat([obs, action], 1)))
        r = self.activ(self.r2(r))
        r = self.r3(r)
        return r

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim,):
        super().__init__()
        
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.ln = nn.LayerNorm(256)
        self.Q01 = nn.Linear(feature_dim + action_shape[0], 256)
        self.Q1 = nn.Sequential(
            nn.Linear(256+repr_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q02 = nn.Linear(feature_dim + action_shape[0], 256)
        self.Q2 = nn.Sequential(
            nn.Linear(256+repr_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action,zsa):
        h = self.trunk(obs)
        h_action = torch.cat([h, action],1)

        h_action1 = self.Q01(h_action)
        h_action_z1 = torch.cat([h_action1,zsa],1)
        q1 = self.Q1(h_action_z1)

        h_action2 = self.Q02(h_action)
        h_action_z2 = torch.cat([h_action2,zsa],1)
        q2 = self.Q2(h_action_z2)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,beta1,beta2,beta3,use_curl):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.use_curl = use_curl
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
        self.cross_entropy_loss = nn.CrossEntropyLoss()
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
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
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
            
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            
            
            fixed_target_zsa = self.fixed_sucencoder_target.zsa(next_obs,next_action)
            
            
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,fixed_target_zsa)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

            fixed_zsa = self.fixed_sucencoder.zsa(obs,action)

        Q1, Q2 = self.critic(obs, action,fixed_zsa)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)


        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        zsa = self.fixed_sucencoder.zsa(obs,action)
        Q1, Q2 = self.critic(obs, action,zsa)
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

    def update_representation(self,obs,obs_pos,action,next_obs,reward,step):
        metrics = dict()
        # 1. CURL loss
        if self.use_curl:
            logits = self.sucencoder.compute_logits(obs,obs_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)

        # 2. zsa loss
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs, stddev)
            policy_action = dist.sample(clip=self.stddev_clip)
            
        pred_zs = self.sucencoder.zsa(obs,action)
        policy_zsa = self.sucencoder.zsa(obs,policy_action)
        
        zsa_loss = F.mse_loss(pred_zs,next_obs) + self.beta1*F.mse_loss(pred_zs.detach(),policy_zsa)
        
        # 3. reward prediction loss
        r = self.sucencoder.r(obs,action)
        r_loss =  F.mse_loss(r,reward)

        repr_loss = zsa_loss + self.beta2*r_loss + self.beta3*curl_loss

        self.encoder_opt.zero_grad()
        self.sucencoder_opt.zero_grad()
        repr_loss.backward()
        self.encoder_opt.step()
        self.sucencoder_opt.step()
        if self.use_tb:
            metrics['zsa_loss'] = zsa_loss.item()
            metrics['r_loss'] = r_loss.item()
            metrics['repr_loss'] = repr_loss.item()
        return metrics


        

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs,\
            sg_obs, sg_action,sg_reward, sg_next_obs = utils.to_torch(
            batch, self.device)
    

        # augment
        obs_aug = self.aug(obs.float())
        next_obs_aug = self.aug(next_obs.float())

        sg_obs_aug = self.aug(sg_obs.float())
        sg_next_obs_aug = self.aug(sg_next_obs.float())


        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()


        # encode
        obs = self.encoder(obs_aug)
        with torch.no_grad():
            next_obs = self.encoder(next_obs_aug)

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))
        # # udpdate encoder
        en_sg_obs = self.encoder(sg_obs_aug)
        sg_obs_aug_pos = self.aug(sg_obs.float())
        en_sg_obs_pos = self.encoder(sg_obs_aug_pos)
        with torch.no_grad():
            en_sg_next_obs = self.encoder(sg_next_obs_aug)

        metrics.update(
            self.update_representation(en_sg_obs,en_sg_obs_pos.detach(),sg_action,en_sg_next_obs,sg_reward,step)
        )


        # # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        utils.soft_update_params(self.fixed_sucencoder, self.fixed_sucencoder_target,
                                 self.critic_target_tau)    
        utils.soft_update_params(self.sucencoder, self.fixed_sucencoder,
                                 self.critic_target_tau)
        return metrics