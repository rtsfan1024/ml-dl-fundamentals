import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# ==========================================
# 第一部分：环境定义 (The Environment)
# 也就是 AI 训练时的“模拟器”或“世界规则”
# ==========================================
@dataclass
class TLConfig:
    road_len: int = 8          # 道路总长度（位置0到8），越过8即为到达终点
    light_pos: int = 5         # 红绿灯所在的具体位置
    cycle_red: int = 20        # 红灯持续的时间（单位：tick/步数）
    cycle_green: int = 20      # 绿灯持续的时间（单位：tick/步数）
    max_steps: int = 60        # 每一轮游戏的步数上限，超时就算失败
    gamma: float = 0.95        # 折扣因子：决定 AI 是短视（只看眼前奖励）还是远视（考虑未来收益）

class TrafficLightEnv:
    # 动作空间 (Action Space)：AI 能做什么？
    # 0 = STOP (原地等待)，1 = GO (往前开一格)
    def __init__(self, cfg=TLConfig()):
        self.cfg = cfg
        self.t = 0             # 当前经过的时间步
        self.pos = 0           # 汽车当前的位置
        self.done = False      # 游戏是否结束的标志

    def reset(self):
        """每轮训练开始前，重置世界状态"""
        self.t = 0
        self.pos = 0
        self.done = False
        return self._obs()

    def _is_green(self):
        """计算当前时间戳下，是红灯还是绿灯"""
        # 利用取余计算周期（前20步红灯，后20步绿灯）
        phase = self.t % (self.cfg.cycle_red + self.cfg.cycle_green)
        return phase >= self.cfg.cycle_red

    def step(self, a):
        """
        核心物理引擎：AI 执行动作 a 后，世界发生什么变化？
        这里定义了强化学习最重要的【奖励函数 (Reward Function)】
        """
        assert not self.done
        reward = 0.0

        if a == 1:  # 如果 AI 选择 GO (前进)
            # 【致命惩罚】：如果刚好在红绿灯前（light_pos - 1），并且是红灯，居然还敢往前开！
            if (self.pos >= self.cfg.light_pos - 1) and (not self._is_green()):
                reward -= 10.0  # 闯红灯，扣大分！(AI 会因此感到极其“痛苦”)
            else:
                self.pos += 1   # 正常安全行驶，位置 +1
        else:       # 如果 AI 选择 STOP (刹车等待)
            # 【轻微惩罚】：停着不动浪费时间，每次扣一点点分，逼迫 AI 尽量往前走
            reward -= 0.1  

        # 【终极通关奖励】：冲过终点线
        if self.pos >= self.cfg.road_len:
            reward += 5.0
            self.done = True

        self.t += 1
        # 超时强行结束游戏
        if self.t >= self.cfg.max_steps:
            self.done = True

        # 返回：下一步能看到的画面(状态), 刚刚动作得分(奖励), 是否结束, 调试信息
        return self._obs(), reward, self.done, {}

    def _obs(self):
        """
        状态空间 (State Space)：AI 能看到什么？
        这里 AI 只能感知两个数值：(自己当前的位置, 现在是红灯还是绿灯)
        """
        return (self.pos, 1 if self._is_green() else 0)

    @property
    def n_states(self):
        """总共有多少种可能的状态组合？位置总数 * 信号灯2种状态"""
        return (self.cfg.road_len + 1) * 2

    @property
    def n_actions(self):
        """总共2个动作：STOP 和 GO"""
        return 2

    def s2i(self, s):
        """将二维状态 (pos, light) 压缩成一个一维的整数索引，方便去查 Q 表"""
        pos, lg = s
        return pos * 2 + lg

# ==========================================
# 第二部分：Q-Learning 算法 (The AI Brain)
# ==========================================
def train_q(env, episodes=320, alpha=0.5, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.985):
    """
    episodes: 训练轮数（玩多少局）
    alpha: 学习率（对新经验的接受程度）
    gamma: 折扣因子（未来的奖励在现在看来值多少钱）
    eps (Epsilon): 探索概率（瞎蒙的概率）
    """
    # 核心大脑：Q表 (Q-Table)。
    # 行 = 所有可能的状态，列 = 所有动作 (GO/STOP)
    # 里面存的数字，代表“在这个状态下，做这个动作，未来一共能拿多少分”
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    
    eps = eps_start # 一开始 eps=1.0，代表 100% 瞎蒙乱开
    
    for _ in range(episodes):
        s = env.reset()
        while True:
            si = env.s2i(s)
            
            # 【Epsilon-Greedy 策略：探索 vs 利用】
            if np.random.rand() < eps:
                # 探索模式：像无头苍蝇一样随机选动作，为了发现新大陆（比如碰巧冲过终点）
                a = np.random.randint(env.n_actions)
            else:
                # 利用模式：查阅 Q 表，看看在这个状态下，选哪个动作历史得分最高
                a = int(np.argmax(Q[si]))
                
            # AI 在环境里实际执行这个动作，看看后果
            ns, r, done, _ = env.step(a)
            ni = env.s2i(ns)
            
            # 【敲黑板！强化学习最核心的 贝尔曼更新公式 (Bellman Equation)】
            # Q(老状态, 刚才的动作) = (1 - 学习率) * 以前的经验 + 学习率 * (刚刚拿到的分 + 远见 * 未来状态能拿到的最高分)
            Q[si, a] = (1 - alpha) * Q[si, a] + alpha * (r + gamma * np.max(Q[ni]))
            
            s = ns # 状态推进
            if done:
                break
                
        # 每一局游戏结束，降低瞎蒙的概率。AI 越来越有经验，越来越倾向于查表
        eps = max(eps_end, eps * eps_decay)
        
    return Q # 训练结束，交出这本写满经验的“秘籍” (Q表)

# ==========================================
# 第三部分：可视化渲染器 (Visualizer)
# 用来把你炼好的 AI 实际开车的样子画出来
# ==========================================
class TLRenderer:
    def __init__(self, env, Q):
        self.env = env
        self.Q = Q
        self.use_learned_policy = True  # g键切换：让 AI 开，还是瞎随机开
        self.paused = False             # 空格键：暂停播放
        self.fig, self.ax = plt.subplots(figsize=(7, 2.4))
        self._setup_canvas()
        self._connect_keys()

    def _setup_canvas(self):
        """画图板初始化，画马路、车、红绿灯和文字说明"""
        cfg = self.env.cfg
        self.ax.plot([0, cfg.road_len + 0.5], [0, 0], linewidth=6)   # 粗线代表道路
        for p in range(cfg.road_len + 1):
            self.ax.plot([p, p], [-0.15, 0.15], linewidth=1)        # 画刻度尺
            
        # 初始化车（一个方块）
        self.car = self.ax.scatter([self.env.pos], [0], s=300, marker="s")
        self.car_label = self.ax.text(self.env.pos, -0.45, "Car", ha="center")
        
        # 初始化红绿灯（一个圆点）
        self.light = self.ax.scatter([cfg.light_pos], [0.5], s=400)
        self.light_txt = self.ax.text(cfg.light_pos, 0.85, "Light", ha="center", va="center")
        
        self.ax.text(cfg.road_len + 0.9, 0, "Goal →", va="center")
        self.title = self.ax.set_title("Training loaded. Playing learned policy… (space: pause / n: step / g: toggle / r: reset / q: quit)")
        self.ax.set_xlim(-0.5, cfg.road_len + 1.8)
        self.ax.set_ylim(-1.0, 1.2)
        self.ax.axis("off") # 隐藏原始的坐标轴边框

    def _connect_keys(self):
        """绑定键盘事件监听"""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key == " ":   self.paused = not self.paused            # 空格暂停
        elif event.key == "n": self._step_once(force=True)              # N 键单步调试
        elif event.key == "r": self.env.reset(); self._render_frame()   # R 键重开一局
        elif event.key == "g": self.use_learned_policy = not self.use_learned_policy # G 键切换 AI 模式
        elif event.key == "q": plt.close(self.fig)                      # Q 键退出

    def _select_action(self, s):
        """决定可视化时的动作"""
        if self.use_learned_policy:
            # 查我们训练好的 Q 表，拿走分数最高的那个动作
            return int(np.argmax(self.Q[self.env.s2i(s)]))
        else:
            # 瞎蒙
            return np.random.randint(self.env.n_actions)

    def _render_frame(self):
        """每一帧更新画面元素"""
        cfg = self.env.cfg
        is_green = self.env._is_green()
        self.light.set_color("green" if is_green else "red")
        
        # 移动车子的图标
        self.car.set_offsets([[self.env.pos, 0]])
        self.car_label.set_position((self.env.pos, -0.45))
        
        # 更新顶部状态栏文字
        self.title.set_text(
            f"{'LEARNED' if self.use_learned_policy else 'RANDOM'} | "
            f"t={self.env.t} | pos={self.env.pos} | light={'Green' if is_green else 'Red'} "
            f"(space pause / n step / g toggle / r reset / q quit)"
        )

    def _step_once(self, force=False):
        """执行一个时间步"""
        if self.env.done:
            self.env.reset()
        s = self.env._obs()
        # 获取动作 -> 丢进环境 -> 更新画面
        a = self._select_action(s) if (force or not self.paused) else 0  
        self.env.step(a)
        self._render_frame()

    def anim_update(self, frame_idx):
        """Matplotlib 动画回调函数，一直循环调用"""
        if not self.paused:
            self._step_once()
        return []

# ==========================================
# 第四部分：主函数入口
# ==========================================
def main():
    cfg = TLConfig()
    env_for_train = TrafficLightEnv(cfg)
    
    # 1. 训练阶段：让 AI 自己玩 320 局，经历无数次闯红灯扣分后，总结出一本 Q 表
    Q = train_q(env_for_train, episodes=320, alpha=0.5, gamma=cfg.gamma)

    # 2. 考试阶段：新建一个干净的环境用于展示
    env = TrafficLightEnv(cfg)
    env.reset()

    # 3. 启动动画渲染器，拿着练好的 Q 表开始表演
    renderer = TLRenderer(env, Q)
    # interval=150 意思是每隔 150 毫秒走一步，你可以改小它让动画播放变快
    ani = FuncAnimation(renderer.fig, renderer.anim_update, interval=150, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()