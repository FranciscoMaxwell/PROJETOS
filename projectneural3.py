# multi_cart_ai_hardcore.py
# Multi-agent RL - versão hardcore com checkpoints dinâmicos

import os, random, math, argparse
from collections import deque, namedtuple
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------
# Hyperparâmetros
# -----------------------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 20000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 20000
MAX_EPISODES = 2000
MAX_STEPS = 800
SAVE_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_AGENTS = 5
N_CHECKPOINTS = 3

# -----------------------
# Replay Buffer
# -----------------------
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self,*args):
        self.buffer.append(Transition(*args))
    def sample(self,batch_size):
        batch = random.sample(self.buffer,batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)

# -----------------------
# DQN
# -----------------------
class DQN(nn.Module):
    def __init__(self,input_dim,output_dim,hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,output_dim)
        )
    def forward(self,x):
        return self.net(x)

# -----------------------
# Ambiente hardcore
# -----------------------
class MultiTrackEnv:
    def __init__(self, screen_size=(640,480), render=True, n_agents=1):
        self.width,self.height = screen_size
        self.render = render
        self.n_agents = n_agents
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("Carrinhos Hardcore")
            self.clock = pygame.time.Clock()
        self.track_rect = pygame.Rect(50,50,self.width-100,self.height-100)
        self.obstacles = [
            pygame.Rect(self.width//2-60,self.height//2-100,40,200),
            pygame.Rect(self.width//2+40,self.height//2-200,40,120)
        ]
        self.checkpoints = []
        self.reset()

    def reset(self):
        self.car_pos=[]
        self.car_angle=[]
        self.speed=[]
        self.time=0
        self.done=[False]*self.n_agents
        for _ in range(self.n_agents):
            self.car_pos.append(np.array([self.track_rect.left+40.0,self.track_rect.centery],dtype=float))
            self.car_angle.append(0.0)
            self.speed.append(0.0)
        # Checkpoints aleatórios
        self.checkpoints=[]
        for _ in range(N_CHECKPOINTS):
            self.checkpoints.append(np.array([random.randint(self.track_rect.left+50,self.track_rect.right-20),
                                              random.randint(self.track_rect.top+50,self.track_rect.bottom-20)],dtype=float))
        return [self._get_state(i) for i in range(self.n_agents)]

    def step(self,actions):
        rewards=[]
        next_states=[]
        dones=[]
        # Movimentação de checkpoints (pequeno desvio aleatório)
        for i in range(len(self.checkpoints)):
            self.checkpoints[i]+=np.random.randn(2)*0.5
            self.checkpoints[i][0]=np.clip(self.checkpoints[i][0],self.track_rect.left+20,self.track_rect.right-20)
            self.checkpoints[i][1]=np.clip(self.checkpoints[i][1],self.track_rect.top+20,self.track_rect.bottom-20)
        for i, action in enumerate(actions):
            if self.done[i]:
                rewards.append(0.0)
                next_states.append(self._get_state(i))
                dones.append(True)
                continue
            turn_speed=6.0
            accel=0.4
            if action==0: self.car_angle[i]-=turn_speed; self.speed[i]+=accel
            elif action==1: self.car_angle[i]-=turn_speed
            elif action==2: self.speed[i]+=0.05
            elif action==3: self.car_angle[i]+=turn_speed
            elif action==4: self.car_angle[i]+=turn_speed; self.speed[i]+=accel
            self.speed[i]=max(-1.0,min(self.speed[i],5.0))
            self.speed[i]*=0.995
            rad=math.radians(self.car_angle[i])
            vx=math.cos(rad)*self.speed[i]
            vy=math.sin(rad)*self.speed[i]
            self.car_pos[i][0]+=vx
            self.car_pos[i][1]+=vy
            self.time+=1
            reward=0.0
            done=False
            # colisão track
            if not self.track_rect.collidepoint(self.car_pos[i][0],self.car_pos[i][1]):
                reward=-100.0
                done=True
            car_rect=pygame.Rect(self.car_pos[i][0]-8,self.car_pos[i][1]-8,16,16)
            for ob in self.obstacles:
                if ob.colliderect(car_rect):
                    reward=-100.0
                    done=True
                    break
            # recompensa pelos checkpoints
            for cp in self.checkpoints:
                dist=np.linalg.norm(self.car_pos[i]-cp)
                reward+=5.0/(dist+1.0)
                if dist<10:
                    reward+=50
            # pequena recompensa por movimento
            reward+=0.5
            if self.time>=MAX_STEPS: done=True
            rewards.append(reward)
            dones.append(done)
            self.done[i]=done
            next_states.append(self._get_state(i))
        return next_states,rewards,dones,{}

    def _get_state(self,idx):
        rays=7
        max_dist=120.0
        angles=np.linspace(-90,90,rays)+self.car_angle[idx]
        dists=[]
        for a in angles:
            rad=math.radians(a)
            dists.append(self._ray_distance(self.car_pos[idx],rad,max_dist)/max_dist)
        speed_norm=self.speed[idx]/5.0
        ang_norm=math.sin(math.radians(self.car_angle[idx]))
        # informações de todos os checkpoints
        cp_dx = [(cp[0]-self.car_pos[idx][0])/self.width for cp in self.checkpoints]
        cp_dy = [(cp[1]-self.car_pos[idx][1])/self.height for cp in self.checkpoints]
        state=np.array(dists+[speed_norm,ang_norm]+cp_dx+cp_dy,dtype=np.float32)
        return state

    def _ray_distance(self,pos,rad,max_d):
        x,y=pos
        step=2.0
        dist=0.0
        while dist<max_d:
            nx=x+math.cos(rad)*dist
            ny=y+math.sin(rad)*dist
            if not self.track_rect.collidepoint(nx,ny):
                return dist
            for ob in self.obstacles:
                if ob.collidepoint((nx,ny)):
                    return dist
            dist+=step
        return max_d

    def render_env(self):
        if not self.render: return
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                pygame.quit()
                raise SystemExit()
        self.screen.fill((30,30,30))
        pygame.draw.rect(self.screen,(50,50,50),self.track_rect)
        for cp in self.checkpoints:
            pygame.draw.circle(self.screen,(30,120,30),(int(cp[0]),int(cp[1])),8)
        for ob in self.obstacles:
            pygame.draw.rect(self.screen,(150,50,50),ob)
        for i in range(self.n_agents):
            x,y=int(self.car_pos[i][0]),int(self.car_pos[i][1])
            pygame.draw.circle(self.screen,(200,200,50),(x,y),8)
            rad=math.radians(self.car_angle[i])
            hx=int(x+math.cos(rad)*8*2.5)
            hy=int(y+math.sin(rad)*8*2.5)
            pygame.draw.line(self.screen,(255,0,0),(x,y),(hx,hy),2)
        pygame.display.flip()
        self.clock.tick(60)

# -----------------------
# Funções de treino e jogar
# -----------------------
def train(render=True):
    env=MultiTrackEnv(render=render,n_agents=N_AGENTS)
    state_dim=len(env._get_state(0))
    n_actions=5
    policy_nets=[DQN(state_dim,n_actions).to(DEVICE) for _ in range(N_AGENTS)]
    target_nets=[DQN(state_dim,n_actions).to(DEVICE) for _ in range(N_AGENTS)]
    for i in range(N_AGENTS): target_nets[i].load_state_dict(policy_nets[i].state_dict())
    optimizers=[optim.Adam(pn.parameters(),lr=LR) for pn in policy_nets]
    buffers=[ReplayBuffer(BUFFER_SIZE) for _ in range(N_AGENTS)]
    # --- Carregar checkpoint se existir ---
    for i in range(N_AGENTS):
        last_path=f"models/agent_{i+1}_last.pth"
        if os.path.exists(last_path):
            policy_nets[i].load_state_dict(torch.load(last_path))
            target_nets[i].load_state_dict(torch.load(last_path))
            print(f"[Checkpoint] Agente {i+1} carregou último modelo!")

    eps=EPS_START
    all_rewards=[[] for _ in range(N_AGENTS)]
    plt.ion()
    fig,ax=plt.subplots()
    lines=[ax.plot([],[],label=f'Agent {i+1}')[0] for i in range(N_AGENTS)]
    ax.set_xlim(0,MAX_EPISODES)
    ax.set_ylim(-10,400)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.legend()

    for ep in range(MAX_EPISODES):
        states=env.reset()
        total_rewards=[0.0]*N_AGENTS
        for step in range(MAX_STEPS):
            actions=[]
            for i,state in enumerate(states):
                if random.random()<eps: actions.append(random.randint(0,n_actions-1))
                else:
                    s_t=torch.tensor(state,dtype=torch.float32).to(DEVICE)
                    qvals=policy_nets[i](s_t)
                    actions.append(int(torch.argmax(qvals).item()))
            next_states,rewards,dones,_=env.step(actions)
            for i in range(N_AGENTS):
                buffers[i].push(states[i],actions[i],rewards[i],next_states[i],dones[i])
                total_rewards[i]+=rewards[i]
            states=next_states
            if render: env.render_env()
            # treino
            for i in range(N_AGENTS):
                if len(buffers[i])>=MIN_REPLAY_SIZE:
                    batch=buffers[i].sample(BATCH_SIZE)
                    state_b=torch.tensor(batch.state,dtype=torch.float32).to(DEVICE)
                    action_b=torch.tensor(batch.action).to(DEVICE)
                    reward_b=torch.tensor(batch.reward,dtype=torch.float32).to(DEVICE)
                    next_state_b=torch.tensor(batch.next_state,dtype=torch.float32).to(DEVICE)
                    done_b=torch.tensor(batch.done,dtype=torch.float32).to(DEVICE)
                    qvals=policy_nets[i](state_b).gather(1,action_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q=target_nets[i](next_state_b).max(1)[0]
                    expected=reward_b+GAMMA*(1-done_b)*next_q
                    loss=nn.MSELoss()(qvals,expected)
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
            if step % TARGET_UPDATE_FREQ==0:
                for i in range(N_AGENTS): target_nets[i].load_state_dict(policy_nets[i].state_dict())
            if all(dones): break
        eps=max(EPS_END,EPS_START*(1-ep/EPS_DECAY))
        for i in range(N_AGENTS):
            all_rewards[i].append(total_rewards[i])
            lines[i].set_data(range(len(all_rewards[i])),all_rewards[i])
        fig.canvas.draw()
        fig.canvas.flush_events()
        if ep%10==0:
            print(f"Episode {ep}: Rewards {[round(r,1) for r in total_rewards]}")
        # Salvamento contínuo
        os.makedirs("models",exist_ok=True)
        for i in range(N_AGENTS):
            torch.save(policy_nets[i].state_dict(),f"models/agent_{i+1}_last.pth")
            if ep%SAVE_EVERY==0:
                torch.save(policy_nets[i].state_dict(),f"models/agent_{i+1}_ep{ep}.pth")
    plt.ioff()
    plt.show()

def play():
    env=MultiTrackEnv(render=True,n_agents=N_AGENTS)
    state_dim=len(env._get_state(0))
    n_actions=5
    policy_nets=[DQN(state_dim,n_actions).to(DEVICE) for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        model_path=f"models/agent_{i+1}_last.pth"
        if os.path.exists(model_path):
            policy_nets[i].load_state_dict(torch.load(model_path))
        else:
            print(f"Modelo do agente {i+1} não encontrado!")
            return
    states=env.reset()
    done_flags=[False]*N_AGENTS
    while not all(done_flags):
        actions=[]
        for i,state in enumerate(states):
            s_t=torch.tensor(state,dtype=torch.float32).to(DEVICE)
            qvals=policy_nets[i](s_t)
            actions.append(int(torch.argmax(qvals).item()))
        states,rewards,done_flags,_=env.step(actions)
        env.render_env()

# -----------------------
# Main
# -----------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--play",action="store_true")
    parser.add_argument("--no-render",action="store_true")
    args=parser.parse_args()
    render=not args.no_render
    if args.train:
        train(render=render)
    elif args.play:
        play()
    else:
        print("Use --train para treinar ou --play para jogar")
