# Automatic Options
I'm going to try and create an automatic options framework here. Not sure it'll work, but it's worth a shot.


___
## Classes
I need three classes for this to work: Energy-class, DQN-class, and something to put it together.


How about to make it simple, I fix the architectures of the DQNs and the AutoEncoders. Why now, it's just a test, and it'll be easy to switch up later.
___

#### Energy
Takes in a list of layer-sizes, among other things. Outputs an energy. Has methods that return the energy, that train it, and that output a list of variables in the model. Give it a random scope name, that should do the trick.

#### DQN
Input is the state, output is the actions. Takes in an energy function, that takes in the state and returns a scalar value. Has methods that get the energy of the state, that train it, that return move-probabilities, that returns the best move with probability 1-eps, and that return a list of variables in the model.

#### The Glue
This still isn't related to the true value function. It's just a way to train a lot of these in succession. Has something like an array of (Energy, DQN) tuples. Can call a "train_next" function, that creates an energy function that's the negative-sum of all the other ones, then trains a DQN based on this, and then trains an energy-model based on the newly-trained model.

