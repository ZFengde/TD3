import math
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from stable_baselines3.common.distributions import Distribution

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=th.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return th.tensor(betas, dtype=dtype)

def extract(a, t, x_shape): # alpha, t, x
    b, *_ = t.shape # get batch number
    out = a.gather(-1, t) # chose value based on the given indexes: t, i.e., out = alpha(t)
    output = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return output # reshape alpha(t) into the shame shape: batch, x_shape

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = t[:,  None] * emb[None, :] 
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Networks(nn.Module): # This network is for generating x_t-1 or epsilon_theta

    def __init__(self,
                 action_dim,
                 state_feat_dim,
                 n_actions,
                 time_dim=16):

        super(Networks, self).__init__()

        self.action_dim = action_dim
        self.state_feat_dim = state_feat_dim
        self.n_actions = n_actions

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Tanh(),
            nn.Linear(time_dim * 2, time_dim),
        )

        inputime_dim = self.state_feat_dim + self.action_dim + time_dim
        self.mid_layer = nn.Sequential(nn.Linear(inputime_dim, 256),
                                       nn.Tanh(),
                                       nn.Linear(256, 256),
                                       nn.Tanh(),
                                       nn.Linear(256, 256),
                                       nn.Tanh())

        self.final_layer = nn.Linear(256, self.action_dim)

    def initialization(self):
        nn.init.xavier_uniform_(self.time_mlp, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.mid_layer, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.final_layer, gain=nn.init.calculate_gain('tanh'))

    def forward(self, action, time, state):
        '''
        shape:
            action: n_env, n_action, action_dim
            state: n_env, 1, state_feat_dim
            time: n_env, n_action, time_dim
        '''
        if action.dim()==3:
            t_embed = self.time_mlp(time).unsqueeze(1).expand(-1, self.n_actions, -1) # time embedding
        else:
            t_embed = self.time_mlp(time)

        x = th.cat([action, t_embed, state], dim=-1).float() # action = 6, 100, 1
        action = self.mid_layer(x)

        return self.final_layer(action)

class Diffusion_Policy(nn.Module): # forward method here is generate a sample

    def __init__(self, 
                 state_feat_dim, 
                 action_dim, 
                 model,
                 n_actions=1,
                 n_timesteps=10):
        super(Diffusion_Policy, self).__init__()

        self.state_feat_dim = state_feat_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.model = model(action_dim=self.action_dim, state_feat_dim=self.state_feat_dim, n_actions=self.n_actions)

        betas = linear_beta_schedule(n_timesteps) # beta

        alphas = 1. - betas # alpha_t
        alphas_cumprod = th.cumprod(alphas, axis=0) # alpha_bar_t

        # alphas_prev here
        alphas_cumprod_prev = th.cat([th.ones(1), alphas_cumprod[:-1]]) # alpha_bar_t-1

        self.n_timesteps = int(n_timesteps)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', th.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', th.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', th.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', th.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', th.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             th.log(th.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) # 
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.gaussian_distribution = Normal(0.0, 1.0)

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, time, noise): # get x_0 prediction from x_t and noise

        # x_0 = x_t/th.sqrt(1. / alphas_cumprod)) - th.sqrt(1. / alphas_cumprod - 1)), reverse formula
        return (
                extract(self.sqrt_recip_alphas_cumprod, time, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, time, x_t.shape) * noise 
        ) 

    def q_posterior(self, x_start, x_t, time):
        
        posterior_mean = (
                extract(self.posterior_mean_coef1, time, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, time, x_t.shape) * x_t
        ) # or actually we could use the noise, which is 
        
        # corresponding to function
        # posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)) 
        # posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
                       
        posterior_variance = extract(self.posterior_variance, time, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, time, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, time, state_feat):
        # (batch_size, self.n_actions, self.action_dim)
        # generate prediction of x_0, noise = eta_theta here
        x_recon = self.predict_start_from_noise(x, time=time, noise=self.model(x, time, state_feat)) 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, time=time) # function(1), x_t-1=f(x_t, noise_theta)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, time, state_feat): # action, time_step, state
        b, *_, device = *x.shape, x.device # batch, _, device

        # model_mean is the mean of x_t-1
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, time=time, state_feat=state_feat)

        # here is random noise to be added as
        noise = th.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (time == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise # x_t-1

    def p_sample_loop(self, state_feat, shape):
        device = self.betas.device
        batch_size = shape[0] 

        # (batch_size, self.n_actions, self.action_dim)
        x = th.randn(shape, device=device) # gaussian random with action shape

        for i in reversed(range(0, self.n_timesteps)):
            time = th.full((batch_size, ), i, device=device, dtype=th.long) # fill batch size with i
            x = self.p_sample(x, time, state_feat)

        return x

    def sample(self, state_feat):
        batch_size = state_feat.shape[0] # get batch size
        shape = (batch_size, self.n_actions, self.action_dim) # make output shape same as action shape
        action = self.p_sample_loop(state_feat, shape) # get x from all diffusion steps
        return action.squeeze()

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state_feat, t):
        noise = th.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        eta_theta = self.model(x_noisy, t, state_feat) # eta_theta

        assert noise.shape == eta_theta.shape
        loss = F.mse_loss(eta_theta, noise) 

        return loss

    def loss(self, x, state_feat): 
        batch_size = len(x) # ground truth
        t = th.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state_feat, t)

    def forward(self, state_feat): 
        state_feat = state_feat.unsqueeze(1).expand(-1, self.n_actions, -1)
        actions = self.sample(state_feat)
        return actions # here is the action

# class Q_networks(nn.Module):
    
#     def __init__(self, env, n_actions, hidden_dim=256):
#         super(Q_networks, self).__init__()

#         '''
#         Q-value model here:
#         Input ---> Output
#         n_env, state_feat_dim, n_action, action_dim ---> 
#         ---> n_env, n_action, state_feat_dim + action_dim
#         ---> Q(s, [a]) = n_env, n_action, 1
#         '''
#         self.action_dim = get_action_dim(env.action_space)
#         self.state_feat_dim = get_flattened_obs_dim(env.observation_space)
#         self.n_actions = n_actions
#         self.q1_model = nn.Sequential(nn.Linear(self.state_feat_dim + self.action_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, 1))

#         self.q2_model = nn.Sequential(nn.Linear(self.state_feat_dim + self.action_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, hidden_dim),
#                                       nn.Mish(),
#                                       nn.Linear(hidden_dim, 1))

#     def multi_q_min(self, state, action): # 64, 100, 27 | 64, 7
#         state = state.unsqueeze(1).expand(-1, self.n_actions, -1)
#         x = th.cat([state, action], dim=-1).float() # avoid overrestimate
#         return th.min(self.q1_model(x), self.q2_model(x))

#     def q1(self, state, action): # multi-actions here
#         state = state.unsqueeze(1).expand(-1, self.n_actions, -1)
#         x = th.cat([state, action], dim=-1).float()
#         return self.q1_model(x)

#     def sinlge_q_min(self, state, action): # 
#         x = th.cat([state, action], dim=-1).float()
#         return th.min(self.q1_model(x), self.q2_model(x))

# def exploration_select_action(actions, q_values): # 4, 100, 8 | 4, 100, 1
#     # could change a action selection method
#     action_index = th.argmax(q_values, dim=1).squeeze() # simply select the aciton with maximum q-value
#     selected_actions = th.stack([actions[i, idx, :] for i, idx in enumerate(action_index)], dim=0).cpu().detach()
#     # selected_actions = actions[:, 0, :] # simply choose the first action
    
#     return selected_actions

# def exploitation_select_action(actions, q_values): # 4, 100, 8 | 4, 100, 1
#     # could change a action selection method
#     action_index = th.argmax(q_values, dim=1).squeeze() # simply select the aciton with maximum q-value
#     # This is only for exploitation
#     selected_actions = th.stack([actions[i, idx, :] for i, idx in enumerate(action_index)], dim=0).cpu().detach()
    
#     return selected_actions



