from typing import NewType, Dict, List, Callable, cast

from labml import monit, tracker, logger, experiment
from labml.configs import BaseConfigs, option

# A player $i \in N$ where $N$ is the set of players
Player = NewType('Player', int)
# Action $a$, $A(h) = {a: (h, a) \in H}$ where $h \in H$ is a non-terminal [history](#History)
Action = NewType('Action', str)


class History:
    """
    <a id="History"></a>

    ## History

    History $h \in H$ is a sequence of actions including chance events,
     and $H$ is the set of all histories.

    This class should be extended with game specific logic.
    """

    def is_terminal(self):
        """
        Whether it's a terminal history; i.e. game over.
        $h \in Z$
        """
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        """
        <a id="terminal_utility"></a>
        Utility of player $i$ for a terminal history.
        $u_i(h)$ where $h \in Z$
        """
        raise NotImplementedError()

    def player(self) -> Player:
        """
        Get current player, denoted by $P(h)$, where $P$ is known as **Player function**.

        If $P(h) = c$ it means that current event is a chance $c$ event.
        Something like dealing cards, or opening common cards in poker.
        """
        raise NotImplementedError()

    def is_chance(self) -> bool:
        """
        Whether the next step is a chance step; something like dealing a new card.
        $P(h) = c$
        """
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        """
        Sample a chance when $P(h) = c$.
        """
        raise NotImplementedError()

    def __add__(self, action: Action):
        """
        Add an action to the history.
        """
        raise NotImplementedError()

    def info_set_key(self) -> str:
        """
        Get [information set](#InfoSet) for the current player
        """
        raise NotImplementedError

    def new_info_set(self) -> 'InfoSet':
        """
        Create a new [information set](#InfoSet) for the current player
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


#Could also wipe this out as well
class InfoSet:
    """
    <a id="InfoSet"></a>

    ## Information Set $I_i$
    """

    # Unique key identifying the information set
    key: str
    
    strategy: Dict[Action, float]
    regret: Dict[Action, float]
    
    cumulative_strategy: Dict[Action, float]

    def __init__(self, key: str):
        """
        Initialize
        """
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Actions $A(I_i)$
        """
        raise NotImplementedError()

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """
        Load information set from a saved dictionary
        """
        raise NotImplementedError()

    def to_dict(self):
        """
        Save the information set to a dictionary
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[str, any]):
        """
        Load data from a saved dictionary
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def calculate_strategy(self):
        """
        ## Calculate strategy

        Calculate current strategy using [regret matching](#RegretMatching).
        """
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(regret.values())
        if regret_sum > 0:
            
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        # Otherwise,
        else:
            count = len(list(a for a in self.regret))
            
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        """
        ## Get average strategy

        """
        cum_strategy = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
       
        strategy_sum = sum(cum_strategy.values())
        if strategy_sum > 0:
           
            return {a: s / strategy_sum for a, s in cum_strategy.items()}
        # Otherwise,
        else:
            count = len(list(a for a in cum_strategy))
            
            return {a: 1 / count for a, r in cum_strategy.items()}

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()


class CFR:
   
    info_sets: Dict[str, InfoSet]

    def __init__(self, *,
                 create_new_history: Callable[[], History],
                 epochs: int,
                 n_players: int = 2):
        """
        * `create_new_history` creates a new empty history
        * `epochs` is the number of iterations to train on $T$
        * `n_players` is the number of players
        """
        self.n_players = n_players
        self.epochs = epochs
        self.create_new_history = create_new_history
        # A dictionary for $\mathcal{I}$ set of all information sets
        self.info_sets = {}
        # Tracker for analytics
        self.tracker = InfoSetTracker()

    def _get_info_set(self, h: History):
        """
        Returns the information set $I$ of the current player for a given history $h$
        """
        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        return self.info_sets[info_set_key]

    #TODO: Imlement this function right here according to the cfr algorithm
    def walk_tree(self, h: History, i: Player, pi_1: float, pi_2: float) -> float:
        """
        - h is the current history
        - i is the player we are computing the regret of
        - pi_1 is probability for reaching a given node in a tree (h) for player 1 given their strategy profile
        - pi_2 above for player 2
        - return the expected utility for the history
        """
        # if h is terminal, return the utility of h for player i
        if h.is_terminal():
            return h.terminal_utility(i)
        
        # if its a chance event P(h) = c sample a and go to the next state
        if h.is_chance():
            # same a single outcome and go to the next state
            a = h.sample_chance()
            return self.walk_tree(h + a, i, pi_1, pi_2)
        
        # get information set and set up expected utilities
        info_set = self._get_info_set(h) # return the information set for the given player
        expected_utility = 0 # initialize the expected utility

        info_set_actions = info_set.actions() # return list of actions from the information set
        
        expected_utility_of_actions = {} # dictionary of actions: utility
        for action in info_set_actions:
            expected_utility_of_actions[action] = 0 # initialize to 0
        

        # iterate through all actions
        for action in info_set_actions:
            # calc utility of action by walking tree and calculating the strategy of the action
            # info_set.strategy[action] is the probability of choosing the action given the information set
            if h.player() == i:
                expected_utility_of_actions[action] = self.walk_tree(h + action, i, info_set.strategy[action] * pi_1, pi_2)
            else:
                expected_utility_of_actions[action] = self.walk_tree(h + action, i, pi_1, info_set.strategy[action] * pi_2)
            # update expected utility, based on prob of picking action
            expected_utility += info_set.strategy[action] * expected_utility_of_actions[action]
        
        if h.player() == i:
            for action in info_set_actions:
                # for each action update the regret and the cumulative strategy
                if i == 1:
                    info_set.regret[action] += pi_2 * (expected_utility_of_actions[action] - expected_utility)
                    info_set.cumulative_strategy[action] += pi_1 * info_set.strategy[action]
                else: # i == 2
                    info_set.regret[action] += pi_1 * (expected_utility_of_actions[action] - expected_utility)
                    info_set.cumulative_strategy[action] += pi_2 * info_set.strategy[action]
            
            # recalculate the strategy (probability of picking actions)
            info_set.calculate_strategy()
        
        return expected_utility

    def iterate(self):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # Loop for `epochs` times
        for t in monit.iterate('Train', self.epochs):
            # Walk tree and update regrets for each player
            for i in range(self.n_players):
                self.walk_tree(self.create_new_history(), cast(Player, i), 1, 1)

            # Track data for analytics
            tracker.add_global_step()
            self.tracker(self.info_sets)
            tracker.save()

            # Save checkpoints every $1,000$ iterations
            if (t + 1) % 1_000 == 0:
                experiment.save_checkpoint()

        # Print the information sets
        logger.inspect(self.info_sets)



class InfoSetTracker:
    """
    ### Information set tracker

    This is a small helper class to track data from information sets
    """
    def __init__(self):
        """
        Set tracking indicators
        """
        tracker.set_histogram(f'strategy.*')
        tracker.set_histogram(f'average_strategy.*')
        tracker.set_histogram(f'regret.*')

    def __call__(self, info_sets: Dict[str, InfoSet]):
        """
        Track the data from all information sets
        """
        for I in info_sets.values():
            avg_strategy = I.get_average_strategy()
            for a in I.actions():
                tracker.add({
                    f'strategy.{I.key}.{a}': I.strategy[a],
                    f'average_strategy.{I.key}.{a}': avg_strategy[a],
                    f'regret.{I.key}.{a}': I.regret[a],
                })


class CFRConfigs(BaseConfigs):
    """
    ### Configurable CFR module
    """
    create_new_history: Callable[[], History]
    epochs: int = 1_00_000
    cfr: CFR = 'simple_cfr'


@option(CFRConfigs.cfr)
def simple_cfr(c: CFRConfigs):
    """
    Initialize **CFR** algorithm
    """
    return CFR(create_new_history=c.create_new_history,
               epochs=c.epochs)
