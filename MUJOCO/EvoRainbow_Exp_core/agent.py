import numpy as np

from EvoRainbow_Exp_core import EvoRainbow_Exp_Algs as algs
from scipy.spatial import distance
from scipy.stats import rankdata
from EvoRainbow_Exp_core import replay_memory
from EvoRainbow_Exp_core.parameters import Parameters
import fastrand
import torch
from EvoRainbow_Exp_core import utils
import scipy.signal
import torch.nn as nn
import  math
from scipy import stats
import random
import time
from EvoRainbow_Exp_core.ES import sepCEM

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

class Agent:
    def __init__(self, args: Parameters, env):
        self.args = args; self.env = env

        # Init population
        self.pop = []
        self.buffers = []
        self.all_actors = []
        for _ in range(args.pop_size):
            genetic = algs.GeneticAgent(args)
            self.pop.append(genetic)
            self.all_actors.append(genetic.actor)

        # Init RL Agent
        self.rl_agent = algs.TD3(args)
        self.replay_buffer = utils.ReplayBuffer()
        self.all_actors.append(self.rl_agent.actor)
        self.ounoise = algs.OUNoise(args.action_dim)
        self.CEM = sepCEM(self.pop[0].actor.get_size(), mu_init=self.pop[0].actor.get_params(),
                          sigma_init=args.sigma_init, damp=args.damp,
                          damp_limit=args.damp_limit,
                          pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2,
                          elitism=args.elitism)
        # Population novelty
        self.ns_r = 1.0
        self.ns_delta = 0.1
        self.best_train_reward = 0.0
        self.time_since_improv = 0
        self.step = 1
        self.use_real = 0
        self.total_use = 0
        # Trackers
        self.num_games = 0; self.num_frames = 0; self.iterations = 0; self.gen_frames = None
        self.rl_agent_frames = 0

        self.old_fitness = None
        self.evo_times = 0
        self.previous_eval = 0
        self.RL2EA = False
        self.rl_index = None
        self.evo_total_times = 0
        self.elite = 0
        self.num_two = 0
        self.others = 0





    def evaluate(self, agent: algs.GeneticAgent or algs.TD3, state_embedding_net, is_render=False, is_action_noise=False,
                 store_transition=True, net_index=None, is_random =False, rl_agent_collect_data = False,  use_n_step_return = False,PeVFA=None):
        total_reward = 0.0
        total_error = 0.0
        policy_params = torch.nn.utils.parameters_to_vector(list(agent.actor.parameters())).data.cpu().numpy().reshape([-1])
        #policy_params = np.array(list(agent.actor.parameters())).reshape([-1])
        state = self.env.reset()
        done = False

        state_list = []
        reward_list = []

        action_list = []
        policy_params_list =[]
        n_step_discount_reward = 0.0
        episode_timesteps = 0
        all_state = []
        all_action = []

        first_in = True

        while not done:
            if store_transition:
                self.num_frames += 1; self.gen_frames += 1
                if rl_agent_collect_data:
                    self.rl_agent_frames +=1
            if self.args.render and is_render: self.env.render()
            
            if is_random:
                action = self.env.action_space.sample()
            else :
                action = agent.actor.select_action(np.array(state))
                if is_action_noise:
                    
                    action = (action + np.random.normal(0, 0.1, size=self.args.action_dim)).clip(-1.0, 1.0)
            all_state.append(np.array(state))
            all_action.append(np.array(action))
            # Simulate one step in environment
            next_state, reward, done, info = self.env.step(action.flatten())
            done_bool = 0 if episode_timesteps + 1 == 1000 else float(done)
            total_reward += reward
            n_step_discount_reward += math.pow(self.args.gamma,episode_timesteps)*reward
            state_list.append(state)
            reward_list.append(reward)
            policy_params_list.append(policy_params)
            action_list.append(action.flatten())



            transition = (state, action, next_state, reward, done_bool)
            if store_transition:
                #next_action = agent.actor.select_action(np.array(next_state))
                self.replay_buffer.add((state, next_state, action, reward, done_bool, None ,policy_params))
                #self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)
            episode_timesteps += 1
            state = next_state

            if use_n_step_return:
                if self.args.time_steps <= episode_timesteps:
                    next_action = agent.actor.select_action(np.array(next_state))
                    param = nn.utils.parameters_to_vector(list(agent.actor.parameters())).data.cpu().numpy()
                    param = torch.FloatTensor(param).to(self.args.device)
                    param = param.repeat(1, 1)

                    next_state = torch.FloatTensor(np.array([next_state])).to(self.args.device)
                    next_action = torch.FloatTensor(np.array([next_action])).to(self.args.device)
                    input = torch.cat([next_state, next_action], -1)
                    next_Q1, next_Q2 = PeVFA.forward(input, param)
                    next_state_Q = torch.min(next_Q1, next_Q2).cpu().data.numpy().flatten()
                    n_step_discount_reward += n_step_discount_reward+math.pow(self.args.gamma,episode_timesteps) *next_state_Q[0]
                    break
        if store_transition: self.num_games += 1
       # print("reward_list",np.mean(reward_list))
        return {'n_step_discount_reward':n_step_discount_reward,'reward': total_reward,  'td_error': total_error, "state_list": state_list, "reward_list":reward_list, "policy_prams_list":policy_params_list, "action_list":action_list}


    def rl_to_evo(self, rl_agent: algs.TD3, evo_net: algs.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self):
        epochs = self.args.ns_epochs
        novelties = np.zeros(len(self.pop))
        for _ in range(epochs):
            transitions = self.replay_buffer.sample(self.args.batch_size)
            batch = replay_memory.Transition(*zip(*transitions))

            for i, net in enumerate(self.pop):
                novelties[i] += (net.get_novelty(batch))
        return novelties / epochs

    def train_RL(self, evo_times,all_fitness, state_list_list,reward_list_list, policy_params_list_list,action_list_list):

        if len(self.replay_buffer.storage) >= 5000:
            before_rewards = np.zeros(len(self.pop))

            discount_reward_list_list =[]
            for reward_list in reward_list_list:
                discount_reward_list = discount(reward_list,0.99)
                discount_reward_list_list.append(discount_reward_list)
            state_list_list = np.concatenate(np.array(state_list_list))
            discount_reward_list_list = np.concatenate(np.array(discount_reward_list_list))
            policy_params_list_list = np.concatenate(np.array(policy_params_list_list))
            action_list_list = np.concatenate(np.array(action_list_list))
            pgl, delta,pre_loss,pv_loss,keep_c_loss= self.rl_agent.train(evo_times,all_fitness, self.pop , state_list_list, policy_params_list_list, discount_reward_list_list,action_list_list, self.replay_buffer ,int(self.gen_frames * self.args.frac_frames_train), self.args.batch_size, discount=self.args.gamma, tau=self.args.tau,policy_noise=self.args.TD3_noise,train_OFN_use_multi_actor=self.args.random_choose,all_actor=self.all_actors)
            after_rewards = np.zeros(len(self.pop))
        else:
            before_rewards = np.zeros(len(self.pop))
            after_rewards = np.zeros(len(self.pop))
            delta = 0.0
            pgl = 0.0
            pre_loss = 0.0
            keep_c_loss = [0.0]
            pv_loss = 0.0
        add_rewards = np.mean(after_rewards - before_rewards)
        return {'pv_loss':pv_loss,'bcs_loss': delta, 'pgs_loss': pgl,"current_q":0.0, "target_q":0.0, "pre_loss":pre_loss}, keep_c_loss, add_rewards

    def train(self):
        self.gen_frames = 0
        self.iterations += 1
        start_time = time.time()


        es_params = self.CEM.ask(self.args.pop_size)
        if not self.RL2EA:
            for i in range(self.args.pop_size):
                self.pop[i].actor.set_params(es_params[i])
        else:
            for i in range(self.args.pop_size):
                if i != self.rl_index:
                    self.pop[i].actor.set_params(es_params[i])
                else :
                    es_params[i] = self.pop[i].actor.get_params()

        self.RL2EA = False

        # ========================== EVOLUTION  ==========================
        # Evaluate genomes/individuals
        real_rewards = np.zeros(len(self.pop))
        fake_rewards = np.zeros(len(self.pop))
        MC_n_steps_rewards = np.zeros(len(self.pop))
        state_list_list = []
        
        store_reward_list_list = []
        reward_list_list = []
        policy_parms_list_list =[]
        action_list_list =[]
        
        if self.args.EA and self.rl_agent_frames>=10000:
            self.evo_times +=1

        if self.args.EA and self.rl_agent_frames>=10000:
            self.evo_times +=1
            random_num_num = random.random()
            if random_num_num< self.args.theta:
                for i, net in enumerate(self.pop):
                    for _ in range(self.args.num_evals):
                        episode = self.evaluate(net, None, is_render=False, is_action_noise=False,net_index=i)
                        real_rewards[i] += episode['reward']
                real_rewards /= self.args.num_evals
                all_fitness = real_rewards
            else :
                for i, net in enumerate(self.pop):
                    episode = self.evaluate(net, None, is_render=False, is_action_noise=False,net_index=i,use_n_step_return = True,PeVFA=self.rl_agent.PVN)
                    fake_rewards[i] += episode['n_step_discount_reward']
                    MC_n_steps_rewards[i]  +=episode['reward']
                all_fitness = fake_rewards
        else :
            all_fitness = np.zeros(len(self.pop))

        self.total_use +=1.0

        self.CEM.tell(es_params, all_fitness)

        if self.rl_index is not None:
            self.evo_total_times +=1
            rank = np.argsort(all_fitness)
            assert rank[-1] == np.argmax(all_fitness)
            if rank[-1] == self.rl_index:
                self.elite += 1
            elif rank[-2] == self.rl_index:
                self.num_two += 1
            else :
                self.others +=1

        keep_c_loss = [0.0 / 1000]
        min_fintess = 0.0
        best_old_fitness = 0.0
        temp_reward =0.0

        # Validation test for NeuroEvolution champion
        best_train_fitness = np.max(all_fitness)
        champion = self.pop[np.argmax(all_fitness)]

        test_score = 0
        
        if self.args.EA and self.rl_agent_frames>=10000 and self.num_frames- self.previous_eval > 5000:
            for eval in range(10):
                episode = self.evaluate(champion, None, is_render=True, is_action_noise=False, store_transition=False)
                test_score += episode['reward']
        test_score /= 10.0

        # NeuroEvolution's probabilistic selection and recombination step

        # ========================== TD3 ===========================
        # Collect experience for training
        if self.rl_index is not None:
            best_index = np.argmax(all_fitness)
            print("best index ", best_index , np.max(all_fitness) , " RL index ",self.rl_index, all_fitness[self.rl_index])
            if best_index != self.rl_index and self.args.EA_tau > 0.0:
                # perform soft update
                for param, target_param in zip(self.pop[best_index].actor.parameters(),self.rl_agent.actor.parameters()):
                    target_param.data.copy_(self.args.EA_tau * param.data + (1 -self.args.EA_tau) * target_param.data)
                for param, target_param in zip(self.pop[best_index].actor.parameters(),self.rl_agent.actor_target.parameters()):
                    target_param.data.copy_(self.args.EA_tau* param.data + (1 - self.args.EA_tau) * target_param.data)


        if self.args.RL:
            is_random = (self.rl_agent_frames < 10000)
            episode = self.evaluate(self.rl_agent, None, is_action_noise=True, is_random=is_random,rl_agent_collect_data=True)

            state_list_list.append(episode['state_list'])
            reward_list_list.append(episode['reward_list'])
            policy_parms_list_list.append(episode['policy_prams_list'])
            action_list_list.append(episode['action_list'])

            if self.rl_agent_frames>=10000:
                losses, _, add_rewards = self.train_RL(self.evo_times,all_fitness, state_list_list,reward_list_list,policy_parms_list_list,action_list_list)
            else :
                losses = {'bcs_loss': 0.0, 'pgs_loss': 0.0 ,"current_q":0.0, "target_q":0.0, "pv_loss":0.0, "pre_loss":0.0}
                add_rewards = np.zeros(len(self.pop)) 
        else :
            losses = {'bcs_loss': 0.0, 'pgs_loss': 0.0 ,"current_q":0.0, "target_q":0.0,"pv_loss":0.0, "pre_loss":0.0}

            add_rewards = np.zeros(len(self.pop))

        L1_before_after = np.zeros(len(self.pop))

        # Validation test for RL agent
        testr = 0
        
        if self.args.RL and self.num_frames- self.previous_eval > 5000:
            for eval in range(10):
                RL_stats = self.evaluate(self.rl_agent, None,store_transition=False, is_action_noise=False)
                testr += RL_stats['reward']
            testr /= 10.0
  
        #Sync RL Agent to NE every few steps
        if self.args.EA and self.args.RL and  self.rl_agent_frames>=10000:
           if self.iterations % self.args.rl_to_ea_synch_period == 0:
               # Replace any index different from the new elite
               replace_index = np.argmin(all_fitness)
               self.rl_to_evo(self.rl_agent, self.pop[replace_index])
               self.RL2EA = True
               self.rl_index = replace_index
               print('Sync from RL --> Nevo')
        elite_index = np.argmax(all_fitness)

        self.old_fitness = all_fitness
        # -------------------------- Collect statistics --------------------------
        print("end ",time.time() - start_time)
        
        if self.num_frames- self.previous_eval > 5000:
            log_wandb = True 
            self.previous_eval = self.num_frames
        else :
            log_wandb = False

        if self.evo_total_times > 0 :
            elite_rate = float(self.elite) / float(self.evo_total_times)
            win_rate = float(self.num_two) / float(self.evo_total_times)
            dis_rate = float(self.others) / float(self.evo_total_times)
        else :
            elite_rate = 0
            win_rate = 0
            dis_rate = 0

        return {
            'log_wandb':log_wandb,
            'elite_rate': elite_rate,
            'win_rate': win_rate,
            'dis_rate': dis_rate,
            'min_fintess':min_fintess,
            'best_old_fitness':best_old_fitness,
            'new_fitness':temp_reward,
            'best_train_fitness': best_train_fitness,
            'test_score': test_score,
            'elite_index': elite_index,
            'RL_reward': testr,
            'pvn_loss':losses['pv_loss'],
            'pg_loss': np.mean(losses['pgs_loss']),
            'bc_loss': np.mean(losses['bcs_loss']),
            'current_q': np.mean(losses['current_q']),
            'target_q':np.mean(losses['target_q']),
            'pre_loss': np.mean(losses['pre_loss']),
            'pop_novelty': np.mean(0),
            'before_rewards':all_fitness,
            'add_rewards':add_rewards,
            'l1_before_after':L1_before_after,
            'keep_c_loss':np.mean(keep_c_loss)
        }