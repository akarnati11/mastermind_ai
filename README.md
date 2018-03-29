## mastermind_ai

A few implementations of mastermind solvers that I wrote for fun. Note: mastermind was proved solvable in less than 5 steps by Knuth in 1977 (https://en.wikipedia.org/wiki/Mastermind_(board_game)).

### Pure Random:
Randomly choose 4 colors for a guess, independent of each other and previous guesses. Models a ~Geom((1/6)^4) with an expected number of guesses as 6^4. Does not use any observation information.

### Random without Replacement:
From an initial codeset of 6^4 codes, uniformly draw codes without replacement. By symmetry expected number of guesses is (6^4-1)/2 = 647.5. Does not use any observation information.

### Genetic:
From a randomly initialized guess, randomly modify each index of the previous code for each index, producing 4 new codes with one different index than the previous code. Choose the code with the highest fitness as the new guess, where fitness is a function of red pins and white pins. The algorithm uses a priority queue to store generated codes, and so it always selects the best code. Seems to be highly dependent on initial guess. Empirical average number of guesses in around [50, 400].

### Markov Chain Monte Carlo:
An adaptation of the Metropolis Markov Chain Monte Carlo algorithm. Begin with a randomly initialized guess, and a 1-initialized 4x6 table of relative probabilities where each row stores the probability of that slot being a particular color. Independently for each slot, sample a color from the candidate function and calculate the acceptance probability (max{1,prob(candidate)/prob(current)}). If the candidate is accepted, change the color of the next guess to the candidate color and update the candidate probability by averaging the score of the candidate (a function of the red and white pins) with the previous color score. Every 10 guesses, instead of choosing the candidate or keeping the current, we choose the color with the maximum probability. The reason for this is that the max probability color converges to the correct color faster than the acceptance to that color. The candidate function was chosen to be a truncated, discretized normal centered at the color with the maximum probability, to encourage candidates closer to current best color. Seems to be highly dependent on initial guess as well, and ranges from guesses between [50, 300].

### Q Learning:
An adaptation of Q learning. Begin with a randomly initialized guess, and a 1-initialized 4x6 table of q-values where each row stores the q-value of that slot being a particular color. Update the q-values of each slot using TD learning (alpha=0.5) with the observed reward (function of red and white pins) and sample a new color for each slot proportional to the q-values. Every 10 guesses, choose the color with the maximum q-value, since it can be seen empirically that the maximum q-values converge to the correct code faster than this code would be produced by a proportional sample. By far the most initial-guess-independent method, almost always beats the other methods and ranges in number of guesses between [30,80].


