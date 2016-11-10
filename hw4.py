import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [-1, 1]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0:
            if action == 1:
                return [(1, 0.3, 80), (-1, 0.7, 20)]
            else:
                return [(1, 0.2, 80), (-1, 0.8, 20)]
        else:
            return [(state, 1, 0)]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None.
    # When the probability is 0 for a particular transition, don't include that
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        total, peeked, rest = state
        results = []
        if rest:
            n = sum(rest)
            if action == "Quit":
                results.append(((total, None, None), 1, total))
            elif action == "Peek":
                if not peeked:
                    results = [((total, i, rest), 1.0 * rest[i]/n, -self.peekCost) for i in xrange(len(rest)) if rest[i] > 0]
            elif action == "Take":
                if peeked:
                    reward = 0
                    new_total = total + self.cardValues[peeked]
                    new_rest = rest[:peeked] + (rest[peeked] - 1,) + rest[peeked+1:]
                    # Check if the deck is empty
                    if n == 1:
                        new_rest = None
                        reward = new_total
                    # Check if the threshold is satisfied
                    if new_total > self.threshold:
                        new_rest = None
                        reward = 0
                    results.append(((new_total, None, new_rest), 1, reward))
                else:
                    for (index, num) in enumerate(rest):
                        if num:
                            prob = 1.0 * num / n
                            reward = 0
                            new_total = total + self.cardValues[index]
                            new_rest = rest[:index] + (rest[index] - 1,) + rest[index + 1:]
                            # Check if the deck is empty
                            if n == 1:
                                new_rest = None
                                reward = new_total
                            # Check if the threshold is satisfied
                            if new_total > self.threshold:
                                new_rest = None
                                reward = 0
                            results.append(((new_total, None, new_rest), prob, reward))
        return results
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    return BlackjackMDP(cardValues=[1, 2, 3, 15], multiplicity = 5, threshold = 20, peekCost=1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState:
            V = max((self.getQ(newState, action), action) for action in self.actions(newState))[0]
            Q = self.getQ(state, action)
            minus = self.getStepSize() * (Q - (reward + self.discount * V))
            for f, v in self.featureExtractor(state, action):
                self.weights[f] = self.weights[f] - minus * v
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
smallMDP.computeStates()
# Let the RL algorithm learn the policy
smallRL = QLearningAlgorithm(smallMDP.actions, smallMDP.discount(), identityFeatureExtractor)
util.simulate(smallMDP, smallRL, numTrials = 30000)
smallRL.explorationProb = 0

# Let Value Iteration solve the Problem
VI = util.ValueIteration()
VI.solve(smallMDP)

# Compare the two policies
diff = 0
for state in smallMDP.states:
    # print "State:", state, "Value Iteration:", VI.pi[state], "QLearning:", smallRL.getAction(state)
    if VI.pi[state] != smallRL.getAction(state):
        diff += 1
print "There are ", 100 * diff / float(len(smallMDP.states)), "% of states yielding different actions."

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP.computeStates()

# Let the RL algorithm learn the policy
largeRL = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), identityFeatureExtractor)
util.simulate(largeMDP, largeRL, numTrials = 30000)
largeRL.explorationProb = 0

# Let Value Iteration solve the Problem
VI = util.ValueIteration()
VI.solve(largeMDP)

# Compare the two policies
diff = 0
for state in largeMDP.states:
    # print "State:", state, "Value Iteration:", VI.pi[state], "QLearning:", largeRL.getAction(state)
    if VI.pi[state] != largeRL.getAction(state):
        diff += 1
print "There are ", 100 * diff / float(len(largeMDP.states)), "% of states yielding different actions."

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    # if state == None:
    #     return [((None, action), 1)]
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    features = []
    features.append(((total, action), 1))
    if counts != None:
        presense = []
        for i in range(len(counts)):
            features.append(((i, counts[i], action), 1))
            presense.append(int(counts[i] > 0))
        features.append(((tuple(presense), action), 1))
    return features
    # END_YOUR_CODE

# Do the large MDP using the new feature extractor again.
# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP.computeStates()

# Let the RL algorithm learn the policy
largeRL = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), blackjackFeatureExtractor)
util.simulate(largeMDP, largeRL, numTrials = 30000)
largeRL.explorationProb = 0

# Let Value Iteration solve the Problem
VI = util.ValueIteration()
VI.solve(largeMDP)

# Compare the two policies
diff = 0
for state in largeMDP.states:
    # print "State:", state, "Value Iteration:", VI.pi[state], "QLearning:", largeRL.getAction(state)
    if VI.pi[state] != largeRL.getAction(state):
        diff += 1
print "There are ", 100 * diff / float(len(largeMDP.states)), "% of states yielding different actions."

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

# Run Value Iteration on original MDP and simulate on the newThresholdMDP
VI = util.ValueIteration()
VI.solve(originalMDP)
FixedVI = util.FixedRLAlgorithm(VI.pi)
util.simulate(newThresholdMDP, FixedVI, numTrials = 30000)

# Run Qlearning on newThresholdMDP
newQ = QLearningAlgorithm(newThresholdMDP.actions, newThresholdMDP.discount(), blackjackFeatureExtractor)
util.simulate(newThresholdMDP, newQ, numTrials = 30000)
newQ.explorationProb = 0

# Compare the two algorithms
rewardList1 = util.simulate(newThresholdMDP, FixedVI, numTrials = 100)
print "Average Reward for Fixed Value Iteration:", 1.0 * sum(rewardList1) / len(rewardList1)
rewardList2 = util.simulate(newThresholdMDP, newQ, numTrials = 100)
print "Average Reward for QLearning Algorithm:", 1.0 * sum(rewardList2) / len(rewardList2)
