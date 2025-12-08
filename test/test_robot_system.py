# robot_system.py
import sys, time, random, math
from collections import deque, defaultdict

ACTIONS = ["UP","DOWN","LEFT","RIGHT","STAY"]
DIRS = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1),"STAY":(0,0)}

class GridWorld:
    def __init__(self,w=15,h=11,seed=0,obs_density=0.12,view=5):
        random.seed(seed)
        self.w=w; self.h=h; self.view=view
        self.grid=[["." for _ in range(w)] for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if random.random()<obs_density: self.grid[i][j]="#"
        self.start=(0,0); self.goal=(h-1,w-1)
        self.grid[self.start[0]][self.start[1]]="S"
        self.grid[self.goal[0]][self.goal[1]]="G"
        self.agent=list(self.start)
        self.steps=0
        self.last_progress_step=0
        self.path_len_est=None
    def reset(self):
        self.agent=list(self.start); self.steps=0; self.last_progress_step=0; self.path_len_est=None
        return self.observe({})
    def inb(self,r,c): return 0<=r<self.h and 0<=c<self.w
    def free(self,r,c): return self.inb(r,c) and self.grid[r][c]!="#"
    def observe(self,io):
        r,c=self.agent
        d=self.view//2
        v=[]
        for i in range(r-d,r+d+1):
            row=[]
            for j in range(c-d,c+d+1):
                if not self.inb(i,j): row.append("X")
                elif [i,j]==self.agent: row.append("A")
                else: row.append(self.grid[i][j])
            v.append(row)
        sp=io.get("speech","")
        return {"vision":v,"speech":sp,"pos":tuple(self.agent),"goal":self.goal,"steps":self.steps}
    def step(self,action):
        dr,dc=DIRS[action]
        nr=self.agent[0]+dr; nc=self.agent[1]+dc
        if self.free(nr,nc): self.agent=[nr,nc]
        self.steps+=1
        done=tuple(self.agent)==self.goal
        r=1.0 if done else -0.01
        return self.observe({}), r, done, {}
    def render(self):
        r,c=self.agent
        g=[]
        for i in range(self.h):
            row=[]
            for j in range(self.w):
                if (i,j)==(r,c): row.append("A")
                else: row.append(self.grid[i][j])
            g.append("".join(row))
        print("\n".join(g))

class VisionEncoder:
    def encode(self,vision):
        m={"X":0,"#":1,".":2,"S":3,"G":4,"A":5}
        return [m[ch] for row in vision for ch in row]

class ASR:
    def transcribe(self,speech): return speech.strip()

class TTS:
    def speak(self,text): print("[TTS]",text)

class MultimodalTokenizer:
    def fuse(self,vision_tokens,speech_text):
        st = [ord(x)%97 for x in speech_text.lower()][:16] if speech_text else []
        return [("v",t) for t in vision_tokens]+[("s",t) for t in st]

class ShortTermMemory:
    def __init__(self,cap=64):
        self.buf=deque(maxlen=cap)
        self.kv={}
    def add(self,item): self.buf.append(item)
    def read(self): return list(self.buf)
    def set(self,k,v): self.kv[k]=v
    def get(self,k,default=None): return self.kv.get(k,default)

class FastController:
    def step(self,obs):
        ar,ac=obs["pos"]; gr,gc=obs["goal"]
        vr=1 if gr>ar else (-1 if gr<ar else 0)
        vc=1 if gc>ac else (-1 if gc<ac else 0)
        cand=[]
        if abs(gr-ar)>=abs(gc-ac):
            if vr<0: cand.append("UP")
            if vr>0: cand.append("DOWN")
            if vc<0: cand.append("LEFT")
            if vc>0: cand.append("RIGHT")
        else:
            if vc<0: cand.append("LEFT")
            if vc>0: cand.append("RIGHT")
            if vr<0: cand.append("UP")
            if vr>0: cand.append("DOWN")
        cand.append("STAY")
        return cand[0]

class Planner:
    def plan(self,env,obs,budget_ms=800):
        start=obs["pos"]; goal=obs["goal"]
        t0=time.time()
        q=deque([start]); prev={start:None}; act={}
        while q and (time.time()-t0)*1000<budget_ms:
            r,c=q.popleft()
            if (r,c)==goal: break
            for a,(dr,dc) in DIRS.items():
                if a=="STAY": continue
                nr,nc=r+dr,c+dc
                if env.free(nr,nc) and (nr,nc) not in prev:
                    prev[(nr,nc)]=(r,c); act[(nr,nc)]=a; q.append((nr,nc))
        if goal not in prev: return []
        path=[]; cur=goal
        while cur!=start:
            path.append(act[cur]); cur=prev[cur]
        path.reverse()
        return path

class Router:
    def __init__(self,progress_patience=12,ood_thresh=0.82):
        self.progress_patience=progress_patience
        self.ood_thresh=ood_thresh
        self.last_pos=None
        self.no_progress=0
    def novelty(self,toks):
        v=[t for tag,t in toks if tag=="v"]
        if not v: return 0.0
        uniq=len(set(v))/max(1,len(v))
        return uniq
    def should_escalate(self,obs,toks,user_intent):
        p=obs["pos"]
        if self.last_pos is None: self.last_pos=p
        if p==self.last_pos: self.no_progress+=1
        else: self.no_progress=0; self.last_pos=p
        if user_intent in ("explain","plan","why"): return True
        if self.no_progress>=self.progress_patience: return True
        if self.novelty(toks)>self.ood_thresh: return True
        return False

class Arbiter:
    def fuse(self,fast_action,plan):
        if plan: return plan[0]
        return fast_action

class PolicyDistiller:
    def __init__(self): self.stats=defaultdict(int)
    def update_from(self,plan): 
        if plan: self.stats[plan[0]]+=1

class RobotSystem:
    def __init__(self,env):
        self.env=env
        self.ve=VisionEncoder()
        self.asr=ASR()
        self.tts=TTS()
        self.fuse=MultimodalTokenizer()
        self.mem=ShortTermMemory()
        self.fast=FastController()
        self.planr=Planner()
        self.router=Router()
        self.arb=Arbiter()
        self.distill=PolicyDistiller()
        self.cached_plan=[]
    def step(self,io):
        speech = self.asr.transcribe(io.get("speech",""))
        obs=self.env.observe({"speech":speech})
        vt=self.ve.encode(obs["vision"])
        toks=self.fuse.fuse(vt,speech)
        user_intent="explain" if any(w in speech.lower() for w in ["why","explain","plan"]) else ""
        fast_action=self.fast.step(obs)
        if self.router.should_escalate(obs,toks,user_intent):
            self.cached_plan=self.planr.plan(self.env,obs,budget_ms=1200)
            self.distill.update_from(self.cached_plan)
            if user_intent: self.tts.speak("I am planning a path to the goal.")
        act=self.arb.fuse(fast_action,self.cached_plan)
        if self.cached_plan and act==self.cached_plan[0]: self.cached_plan=self.cached_plan[1:]
        ob2,r,done,_=self.env.step(act)
        self.mem.add({"obs":obs,"act":act,"r":r})
        return ob2,r,done

    def run(self,max_steps=200,interactive=False):
        obs=self.env.reset()
        
        # 智能与控制之间的关系，我要搞清楚里面的逻辑关系，
          
        # method 1: loop inside the robot.
        # for i in range(0, infinite):  
        #    self.sense(env) # sense the input from env. 
        #    self.act(env) # act the output to env. 
        
        # method 2: two process, sense and act processes run independently.
        # process A:
        # for i in range(0, infinite):
        #     self.sense(env)
        # process B:
        # for i in range(0, infinite):
        #     self.act(env)

        # method 3: 开启 SenseStreaming 
        # self.audio_sensor.run_streaming() # 开启audio streaming. 
        # self.vision_sensor.run_streaming() # 开启vision streaming.
        
        # 线程化 module.  
        # self.audio_encoder.run(input=self.audio_sensor.stream, hz=10) # --> project to self.audio_encoder_module.output_stream # with x hz, and embeddings.
        # self.vision_encoder.run(input=self.vision_sensor.stream, hz=10) # 
        # self.robot_status.run(hz=50)
        # self.vlm_encoder.run(input_audio=self.audio_encoder.stream, input_vision=self.vision_encoder.stream, input_status=self.robot_status.stream)
        # self.action.run(input_status=self.robot_status.stream, input_vision=self.vision_encoder.stream, input_semantic=self.vlm_encoder.semantic_stream, hz=50) # system 1, 会拿到新的位置.  
        # self.speaker.run(input_text=self.vlm_encoder.word_stream)  
        #
        # method 3 应该是我们会采用的方法。
        # 这里面核心存在一个问题，如何同时支持 1. 即时响应; 2. reasoning. 比如在 TTS 里面。 简单而言就是通过控制多段<think>来达到这一点。
        
        


        for t in range(max_steps):
            io={}
            if interactive:
                try:
                    io["speech"]=input("You> ")
                except EOFError:
                    io["speech"]=""

            ob2,r,done=self.step(io)
            
            if interactive: print("Action:",self.mem.read()[-1]["act"],"Reward:",r)
            
            if done:
                self.tts.speak("Goal reached.")
                return True
        
        self.tts.speak("Stopped without reaching the goal.")
        
        return False


def demo(non_interactive=True):
    # 仿真环境. 
    env=GridWorld(seed=42)

    # 机器人系统. 
    bot=RobotSystem(env)

    if non_interactive:
        scripted=["","why","explain plan",""]
        k=0
        env.render(); print()
        for _ in range(200):
            sp=scripted[min(k,len(scripted)-1)]; k+=1
            ob2,r,done=bot.step({"speech":sp})
            if done: break
        env.render()
    else:
        bot.run(interactive=True)

if __name__=="__main__":
    ni = "--interactive" not in sys.argv
    demo(non_interactive=ni)
