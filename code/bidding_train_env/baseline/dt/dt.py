"""
Generative Auto-Bidding with Value-Guided Explorations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from gin import current_scope_str


def getScore(budget, cpa_cons, states, all_reward):
    '''
    计算当前时刻的 带惩罚的score S_t (Eq.14)
    score是为了计算 r_t (return-to-go)的: r_t = S_T - S_{t-1}

    budget: 总预算
    states[1]: 预算剩余比例
    cpa_cons: 广告主对CPA的约束
    '''
    gamma = 2
    curr_cost = budget * (1 - states[1])    # 已花预算 = 总预算 * (1-剩余比例)
    curr_all_reward = all_reward        # 价值总和，rw = sun(xivi)
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)    # 当前CPA
    curr_coef = cpa_cons / (curr_cpa + 1e-10)   #
    curr_penalty = pow(curr_coef, gamma)    # 惩罚系数
    curr_penalty = 1.0 if curr_penalty > 1.0 else curr_coef
    curr_score = curr_penalty * curr_all_reward      # 带惩罚的 score

    return curr_score

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        # 下三角 mask，保证因果性
        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()      # batch, seq_len, dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, n_head, T, head_dim)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]   # 变成(B,1,1,T)方便广播
        mask = (1.0 - mask) * -10000.0      # -10000相当于-inf
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T)
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))  # 应用下三角mask
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()    # 可视化用
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    # 解码器层： Attn+FFN
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Dropout(config['resid_pdrop']),
            nn.Linear(config['n_inner'], config['n_embd']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GAVE(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, hidden_size=64, action_tanh=False, K=20,
                 max_ep_len=96, scale=2000, warmup_steps=10000, weight_decay=0.0001,learning_rate=0.0001, time_dim=8,
                 target_return=4, device="cpu", expectile=0.99,
                 block_config={
                     "n_ctx": 1024,
                     "n_embd": 64,
                     "n_layer": 3,
                     "n_head": 1,
                     "n_inner": 512,
                     "activation_function": "relu",
                     "n_position": 1024,
                     "resid_pdrop": 0.1,
                     "attn_pdrop": 0.1,
                 }
                 ):
        super(GAVE, self).__init__()
        self.device = device

        self.length_times = 3
        self.hidden_size = hidden_size
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.time_dim = time_dim
        self.expectile = expectile      # 分位数的 τ

        block_config = block_config

        # Transformer骨干
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        # PE
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)      # 位置t的emb层
        # 嵌入层
        self.embed_return = torch.nn.Linear(1, self.hidden_size)    # r_t（return-to-go）的Embedding层，标量→向量
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)    # rw_t (r_t就是rw_t的return-to-go)，标量→向量
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)    # s的Embedding层
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)     # a的Embedding层
        # 用于加入PE的嵌入层, 映射后 r, rw, s, a 维度都是 hidden_size
        self.trans_return = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_reward = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_state = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_action = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        # layernorm
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # 预测头
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)      # 下一状态（可选），GAVE未使用ŝ_{t+1}
        
        layers = [nn.Linear(self.hidden_size, self.act_dim)]
        if action_tanh:
            layers.append(nn.Tanh())
        self.predict_action = nn.Sequential(*layers)     # â_t的映射，hidden_size -> act_dim

        self.predict_beta = nn.Sequential(      # β_t的映射，hidden_size -> 1
            nn.Linear(self.hidden_size, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        self.predict_return = nn.Sequential(     # r̂_{t+1}的映射，hidden_size -> 1
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.predict_value = nn.Sequential(     # V̂_{t+1}的映射，hidden_size -> 1
            nn.Linear(self.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        """
        一次 forward 完成两件事：
        1. 训练模式：
            先算 â_t、β_t、V̂_{t+1}，
            再构造 ã_t = β_t·a_t，
            再算 r̂_{t+1}(ã_t)
        2. 推理模式：只输出 â_t

        输入长度 = K 步历史，输出长度 = K 个位置级预测（不是下一个 token）
        """
        batch_size, seq_length = states.shape[0], states.shape[1]       # 传进来的就是K步历史

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # ① 嵌入层：把原始标量/向量做emb
        state_embeddings = self.embed_state(states)     # emb s_{t-k:t}
        action_embeddings = self.embed_action(actions)   # emb a_{t-k:t}
        returns_embeddings = self.embed_return(returns_to_go)       # emb r_{t-k:t}（RTG）
        rewards_embeddings = self.embed_reward(rewards)      # emb rw_{t-k:t}（即时奖励）
        time_embeddings = self.embed_timestep(timesteps)       # emb pos_{t-k:t}
        # ② 时间融合：emb向量 + 位置向量 → 融合后向量 → 再做一次mlp
        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        returns_embeddings = torch.cat((returns_embeddings, time_embeddings), dim=-1)
        rewards_embeddings = torch.cat((rewards_embeddings, time_embeddings), dim=-1)

        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        returns_embeddings = self.trans_return(returns_embeddings)
        rewards_embeddings = self.trans_reward(rewards_embeddings)

        # ③ 拼接起来得到序列
        '''
        嵌入向量的堆叠顺序:
        # stack得到 (B, 3, K, d)
        # permute后变为(B, K, 3, d), 再reshape成(B, 3K, d)
        # 即：[r₀, s₀, a₀,  r₁, s₁, a₁,  ...,  r_T, s_T, a_T]
        '''
        stacked_inputs = torch.stack(
            (returns_embeddings,    # 0 Rt
             state_embeddings,      # 1 st
             action_embeddings),    # 2 at
             dim=1    
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size) 
        stacked_inputs = self.embed_ln(stacked_inputs)

        # ④ 构造因果 mask：3K × 3K 下三角，保证 token i 只能看到 ≤i 的信息
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        # ⑤ Transformer 编码 → 输出因果隐向量 (B, 3K, d)
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)
        
        # 经过reshape后 -> (B, K, 3, d)
        # permute -> (B, 3, K, d)
        x = x.reshape(-1, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        # 那么 x[:, 0, :, :] → return 位隐向量 h^r
        # x[:, 1, :, :] → state 位隐向量 h^s
        # x[:, 2, :, :] → action 位隐向量 h^a
        return_preds = self.predict_return(x[:, 2])     # 用at及以前的信息(..., Rt, st, at) 去预测 Rt+1
        state_preds = self.predict_state(x[:, 2])       # 用at及以前的信息(..., Rt, st, at) 去预测 st+1
        action_preds = self.predict_action(x[:, 1])     # 用st及以前的信息(..., Rt, st) 去预测 at

        if self.training:
            # ⑧ 探索动作 ã_t = β_t · a_t（论文 β_t）
            value_preds = self.predict_value(x[:, 1])       # 用st及以前的信息(..., Rt, st) 去预测V̂_{t+1}
            beta_preds = self.predict_beta(x[:, 1]) + 0.    # 用st及以前的信息(..., Rt, st) 去预测β_t ∈ (0.5, 1.5)
            actions_1 = actions.clone().detach() * beta_preds   # ã_t
            # 给 ã_t 做 emb（emb + PE + emb）
            action_embeddings_1 = self.embed_action(actions_1)
            action_embeddings_1 = torch.cat((action_embeddings_1, time_embeddings), dim=-1)
            action_embeddings_1 = self.trans_action(action_embeddings_1)
            
            stacked_inputs_1 = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings_1), dim=1  # [..., Rt-1, st-1, ãt-1, Rt, st, ãt] 感觉这样把a都替换了会污染数据 但可能它觉得问题不大
            ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
            stacked_inputs_1 = self.embed_ln(stacked_inputs_1)
            stacked_attention_mask_1 = torch.stack(
                ([attention_mask for _ in range(self.length_times)]), dim=1     # 下三角mask
            ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs_1.dtype)

            # 放到Transformer里编码
            x_1 = stacked_inputs_1
            for block in self.transformer:
                x_1 = block(x_1, stacked_attention_mask_1)
            x_1 = x_1.reshape(-1, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)
            return_preds_1 = self.predict_return(x_1[:, 2])     # 用at及以前的信息(..., Rt, st, at) 去预测 Rt+1
            return state_preds, action_preds, return_preds, None, return_preds_1, actions_1, value_preds

        return None, action_preds, None, None, None, None, None

    def get_action(self, states, actions, rewards, curr_score, timesteps, **kwargs):
        '''infer时用，调用DT'''
        # --------------- 1. reshape 成 batch=1 ---------------
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        curr_score = curr_score.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # --------------- 2. 滑动窗口 + pad 到固定长度 K ---------------
        if self.max_length is not None:
            # 只留最近的K步（滑动窗口）
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            curr_score = curr_score[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # 构造 attention mask：前面 pad 部分=0，真实部分=1
            # states.shape[1]：已经有数据的历史时间步数
            # 在前面填 0，直到长度填到K
            K = self.max_length
            attention_mask = torch.cat([torch.zeros(K - states.shape[1]), torch.ones(states.shape[1])])       
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], K - states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], K - actions.shape[1], self.act_dim), device=actions.device), actions], dim=1).to(dtype=torch.float32)
            curr_score = torch.cat(
                [torch.zeros((curr_score.shape[0], K - curr_score.shape[1], 1), device=curr_score.device), curr_score], dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], K - rewards.shape[1], 1), device=rewards.device), rewards], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], K - timesteps.shape[1]), device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        
        # ===== 3. forward 一次 =====
        # 一次forward 同时得到 action_preds, curr_score_preds, reward_preds
        # 输出维度 (1, 3K, 64)
        _, action_preds, curr_score_preds, reward_preds, _, _, _ = self.forward(
            states, actions, rewards, curr_score, timesteps, attention_mask=attention_mask, **kwargs)       # action_preds ≡ 论文 â_t 序列 (长度 K)

        # ===== 4. 只拿最后一个动作 =====
        return action_preds[0, -1]  # 维度 (action_dim,)

    def step(self, states, actions, rewards, dones, all_reward, curr_score, timesteps, attention_mask, next_states):
        """
        训练阶段一次 update 的完整 loss 计算：
        1. 先 forward 拿到 â_t、β_t、V̂_{t+1}、r̂_{t+1}(a_t)、r̂_{t+1}(ã_t)
        2. 构造 4 项 loss（动作回归 + RTG 回归 + expectile + OOD 正则）
        3. 梯度更新
        返回：总 loss + 各子 loss + 监控量
        """

        # ① 构造真值标签（detach 避免梯度回传）
        action_target, curr_score_target = torch.clone(actions).detach(), torch.clone(curr_score).detach()      # a_t, R_t
        state_target = torch.clone(next_states).detach()        # s_{t+1}
        curr_score_target = curr_score_target[:, 1:]        # R_t 标签往后错一位 → score_{t+1}（用于 r̂_{t+1} 回归）

        # forward一次，拿到预测的s^_t, â_t、r̂_{t+1}(a_t)、r̂_{t+1}(ã_t)、ã_t、V̂_{t+1}
        state_preds, action_preds, curr_score_preds, _, curr_score_preds_1, action_1, value_preds = self.forward(
            states, actions, rewards, curr_score[:, :-1], timesteps, attention_mask=attention_mask,
        )
        # ③ 只保留非 pad  token（attention_mask=1 才有效）
        # â_t
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # ã_t
        action_1 = action_1.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_1_frozen = action_1.clone().detach()     # 探索动作 ã_t（梯度冻结）
        # s^_t
        state_dim = state_preds.shape[2]
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        # r̂_{t+1}(a_t)
        curr_score_dim = curr_score_preds.shape[2]
        curr_score_preds = curr_score_preds.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        # r̂_{t+1}(ã_t)
        curr_score_preds_1 = curr_score_preds_1.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        curr_score_target = curr_score_target.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        # V̂_{t+1}
        value_preds = value_preds.reshape(-1, curr_score_dim)[attention_mask.reshape(-1) > 0]
        value_preds_frozen = value_preds.clone().detach()    # V̂_{t+1}（梯度冻结）

        # Loss function without learnable value function. It's more stable but may perform worse than a finetuned loss with learnable value function.
        # In this case, we simply boost exploration by maxmaizing curr_score_preds_1 in wo.
        # wo = torch.sigmoid(1 * (curr_score_preds_1-curr_score_preds.clone().detach()))
        # wo_frozen = wo.clone().detach()
        # loss1 = torch.mean((1-wo_frozen)*((action_preds - action_target) ** 2) + wo_frozen*((action_preds - action_1_frozen) ** 2))
        # loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2)*200
        # loss3 = torch.mean(1-wo)*100
        # loss = loss1+loss2+loss3

        # The loss in the paper. It's param sensitive and need careful param selection.

        # ④ 探索权重 w_t
        # wo ≡ w_t = sigmoid( α · (r̂_{t+1}(ã_t) - r̂_{t+1}(a_t)) )
        wo = torch.sigmoid(100 * (curr_score_preds_1 - curr_score_preds))
        wo_frozen = wo.clone().detach()
        # ⑤ expectile 权重（论文 3.4，τ=0.99）
        diff = curr_score_target - value_preds
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        loss1 = torch.mean((1 - wo_frozen) * ((action_preds - action_target) ** 2) +
                           wo_frozen * ((action_preds - action_1_frozen) ** 2))     # 动作回归 + 探索偏向：用ã_t和a_t去训练â_t
        loss2 = torch.mean((curr_score_preds - curr_score_target) ** 2) * 200       # 用真Rt+1训练预测的Rt+1
        loss3 = torch.mean(weight * (diff ** 2)) * 100                              # 分位数上界回归，用Rt+1的分位数去训练Vt+1
        loss4 = torch.mean((curr_score_preds_1 - value_preds_frozen) ** 2) * 100    # 探索动作对应的r̂_{t+1}(ã_t) 向 V̂ 靠拢（OOD 正则）
        loss = loss1 + loss2 + loss3 + loss4
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        # 监控
        return (loss.detach().cpu().item(), loss1.detach().cpu().item(), loss2.detach().cpu().item(), loss3.detach().cpu().item(),
                loss4.detach().cpu().item(), torch.mean(wo_frozen.squeeze()).cpu().item(), torch.mean(curr_score_target).cpu().item(),
                torch.mean(curr_score_preds).cpu().item(), torch.mean(curr_score_preds_1).cpu().item())

    def take_actions(self, state, target_return=None, pre_reward=None, budget=100, cpa=2):
        '''
        infer时的在线推理入口：每收到一条新状态，返回动作â_t
        
        pre_reward: rw_{t-1}
        state: s_t
        target_return: R_T
        '''
        self.eval()

        # --------------- 1. 构造或更新“目标回报R_0，也就是0时刻的return-to-go（整条轨迹的期望总回报）
        target_return = target_return.to(self.device) if target_return is not None else self.target_return
        if self.eval_states is None:
            # ===== 第一次推理（t=0） =====
            # 状态序列 s_{0:t}
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)      
            # 累计即时奖励 Σ_{k=0}^{t-1} rw_k
            all_reward = torch.zeros(1).to(self.device)     
            self.eval_all_reward = torch.tensor(all_reward, dtype=torch.float32).reshape(1, 1).to(self.device)
            # S_{0:t}（带 CPA 惩罚的t时刻的score），初始化为 target_return，即R_T
            self.eval_curr_score = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1).to(self.device)   

        else:
            # ===== 第 2...N 步（t≥1） =====
            # 累计奖励 & 当前score
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)      # s_t（当前这一步收到的新状态st）
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)  # 更新状态序列
            
            self.eval_rewards[-1] = pre_reward      # 把上一步收到的即时奖励 rw_{t-1} 填到缓存
            pred_all_reward = self.eval_all_reward[0, -1] + pre_reward      # 累计即时奖励 ≡ Σ_{k=0}^{t-1} rw_k
            self.eval_all_reward = torch.cat([self.eval_all_reward, pred_all_reward.reshape(1, 1)], dim=1)
            # 计算当前score: R0 - 带惩罚的 score  target_return暂时不知道是R0还是ST
            curr_score = target_return - getScore(budget, cpa, self.eval_states[-1], pred_all_reward) / self.scale
            self.eval_curr_score = torch.cat([self.eval_curr_score, curr_score.reshape(1, 1)], dim=1)
            # 时间步 +1
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps,
                 torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)
        # 给下一步留一个动作占位
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])

        # 拿动作：【关键】整条历史再 forward 一次
        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(
                self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_curr_score.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long)
        )
        # 把预测动作写回缓存（-1 是占位的那一行）
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def init_eval(self):
        '''清空推理缓存'''
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)
        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path, name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, name)
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DTtest/dt.pt", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"Model loaded from {self.device}.")
