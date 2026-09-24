# DDQN_model.py
import torch
from compare.DQN_model import DQN  # 复用你已实现的 DQN（注意其中 get_operations 已改为使用 MLPs）

class DDQN(DQN):
    """
    Double DQN：仅重写目标计算：
      y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
    其它行为（采样、优化、目标网更新频率/软更新）全部沿用 DQN 基类。
    """
    @torch.no_grad()
    def _max_q_next(self, next_state):
        # 1) 在线网络选择动作（带合法掩码，非法动作在 _q_forward 里已置为极小值）
        q_online, mask_next, _ = self.policy._q_forward(next_state, require_grad=False)
        if mask_next.sum() == 0:
            # s' 无任何合法动作，返回 0
            return torch.tensor(0.0, device=self.device)

        a_star = torch.argmax(q_online, dim=1)[0]   # 选 a* = argmax_a Q_online(s', a)

        # 2) 目标网络评估该动作的 Q 值
        q_target, _, _ = self.policy_old._q_forward(next_state, require_grad=False)
        q_val = q_target[0, int(a_star.item())]
        return q_val
