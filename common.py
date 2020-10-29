'''
This script is for the classses, functions that are commonly used among other components.
Like action and observation space definitions
'''


def serialize(self):
    _dict = {}
    for el in self.__dict__:
        att = getattr(self, el)
        _dict[el] = att
    return _dict

class Space:
    def __init__(self, type):
        pass


def calc_returns(rewards, discount=0.9, n=-1):
    '''
    Calculates the returns for each step after a single episode.
    This function can be used for the cases where all the updates are done after a set amount of episodes
    :param rewards: Array of collected rewards at an episode
    :param discount: Discount rate
    :param n: N-step bootstrapping. N=-1 means no bootstrapping
    :return: Array of returns for each step.
    '''
    returns = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r = r + discount * sum_r
        returns.append(sum_r)
    return list(reversed(returns))

def polyak_update(target, source, p=0.5):
    '''
    Updates the parameters of the target network. If p = 0, no update, if p = 1 completely update with the source params
    :param target: Network to update
    :param source: Source network
    :param p: Averaging param
    :return:
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - p) + param.data * p
        )


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True
