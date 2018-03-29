import random
import numpy as np
import scipy.stats as stats
import heapq
import matplotlib.pyplot as plt


def pureRandom(code, trials=100):
	av = None
	for _ in range(trials):
		guess = None
		c = 0
		while guess is None or guess != code:
			guess = [random.randint(0,5) for _ in range(4)]
			c+=1
		if av is None:
			av = c
		else:
			av = 0.5*av + 0.5*c

	print("Pure Random (guesses):", av)

def randomNoReplace(code, trials=100):
	code = sum([code[i]*6**(3-i) for i in range(4)])


	av = None
	for _ in range(trials):
		guess = None
		c = 0
		totalSequences = [i for i in range(6**4)]
		while guess is None or guess != code:
			guess = random.sample(totalSequences, 1)[0]
			totalSequences.remove(guess)
			c+=1
		if av is None:
			av = c
		else:
			av = 0.5*av + 0.5*c
	print("Random w/o Replacement (guesses):", av)


def genetic(code, curGuess, trials=1000):
	av = None
	for _ in range(trials):
		curGuess = initGuess[:]
		pq = []
		scores = [compObs(curGuess, code, 2)]
		c = 0
		while curGuess != code:
			for d in range(4):
				curGuess = [curGuess[i] if i != d % 4 else random.randint(0,5) for i in range(4)]
				
				redPin = sum([curGuess[i] == code[i] for i in range(4)])
				whitePin = sum([1 for i in curGuess if i in code]) - redPin
				obs = compObs(curGuess, code, 2)
				heapq.heappush(pq, (-obs, curGuess))
			c+=4
			curGuess = heapq.heappop(pq)
			curGuess = curGuess[1]
			scores.append(compObs(curGuess, code, 2))
		if av is None:
			av = c
		else:
			av = (av + c)/2
	print("Genetic:", av)
	# print(scores)
	# plt.plot(range(len(scores)), scores)
	# plt.show()


def compObs(g, code, scale=4):
	redPin = sum([g[i] == code[i] for i in range(4)])
	whitePin = sum([1 for i in g if i in code]) - redPin
	return scale*redPin + whitePin

def normalProp(mu, sd):
	prop = int(np.round(np.random.normal(loc=mu, scale=sd)))
	if prop < 0: prop = 0
	elif prop > 5: prop = 5
	return prop


def mcmc(code, initGuess, trials=1000):
	av = None
	for _ in range(trials):
		probs = [[1 for _ in range(6)] for __ in range(4)]
		states = [initGuess]
		curGuess = initGuess[:]
		c = 0
		while curGuess != code:
			curProp = []
			for i in range(4):
				prop = normalProp(np.argmax(probs[i]), 1.5)
				# print(prop)
				if random.random() <= probs[i][prop]/probs[i][curGuess[i]]:
					curGuess[i] = prop
					probs[i][curGuess[i]] = (compObs(curGuess, code) + probs[i][curGuess[i]])/2
				curProp.append(curGuess[i])
			if c % 10 == 0:
				curGuess[i] = np.argmax(probs[i])

			states.append(curProp)
			c+=1
	if av is None:
		av = c
	else:
		av = (av + c)/2
	print("MCMC Method:", av)
	states = np.array(states)
	# for i in range(4):
	# 	plt.plot(range(len(states)), states[:,i])
	# 	plt.plot(range(len(states)), np.full((len(states),), np.mean(states[:,i])))
	# 	plt.show()


def sample(cP):
	r = random.random()
	cSum = 0
	for i in range(6):
		cSum += cP[i]/sum(cP)
		if r <= cSum:
			return i
	return -1


def reinforcement(code, initGuess, trials=1000):
	av = None
	alpha = .5
	for _ in range(trials):
		Q = [[1 for _ in range(6)] for __ in range(4)]
		curGuess = initGuess[:]
		c = 0

		while curGuess != code:
			obs = compObs(curGuess, code)

			for i in range(4):
				Q[i][curGuess[i]] = (1-alpha)*Q[i][curGuess[i]] + alpha*obs

			if c % 10 != 0:
				curGuess = [sample(q) for q in Q]
			else:
				curGuess = [np.argmax(q) for q in Q]

			c += 1
			# print(Q)

		if av is None:
			av = c
		else:
			av = (av + c) / 2
	print("Q Learning:", av)


code = [random.randint(0,5) for _ in range(4)]
print("Code:", code)

initGuess = [random.randint(0,5) for _ in range(4)]
print("Init Guess:", initGuess)

pureRandom(code)
randomNoReplace(code)
genetic(code, initGuess)
mcmc(code, initGuess)
reinforcement(code, initGuess)

