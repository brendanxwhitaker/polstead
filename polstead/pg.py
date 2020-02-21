""" An implementation of a simple policy gradient agent. """
from asta import Array
from torch.distributions.categorical import Categorical

class Trajectories:
    """ An objection to hold trajectories of numpy arrays. """
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.action_log_probs: List[np.ndarray] = []
        self.rewards: List[float] = []


class ActorCritic:
    """ The parameterized action-value and state-value functions. """
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        # Actor implements the policy $\pi_{\theta}$.
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1),
        )

        # Critic estimates the on-policy value function $V^{\pi_{\theta}}(s)$.
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def get_policy(self, ob: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ Computes the policy distribution for the state given by ``ob``. """
        logits: torch.Tensor = self.actor(ob) 
        policy = Categorical(logits=logits)
        return policy
