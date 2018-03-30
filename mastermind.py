import random
import numpy as np
import heapq
import scipy.stats as stats
import matplotlib.pyplot as plt
import operator


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
				
				obs = compObs(curGuess, code)
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

def compScore(g, p, code):
	gRP = sum([g[i] == code[i] for i in range(4)])
	gWP = sum([1 for i in g if i in code]) - gRP

	pRP = sum([p[i] == g[i] for i in range(4)])
	pWP = sum([1 for i in p if i in g]) - pRP

	gRAbs = abs(gRP - 4)
	gWAbs = abs(gWP - 0)

	pRAbs = abs(pRP - gRP)
	pWAbs = abs(pWP - gWP)

	tAbs = abs(gRP + gWP - pRP - pWP)

	return np.e**(-2*(gRAbs + gWAbs + 2*tAbs)), np.e**(-2*(pRAbs + pWAbs + 2*tAbs))

def candCode(g):
	if random.random() < 0.5:
		choices = [0,1,2,3]
		i1 = np.random.choice(choices, replace=False)
		i2 = np.random.choice(choices)
		g[i1], g[i2] = g[i2], g[i1]
		return g
	i = random.randint(0,3)
	return [g[j] if j != i else random.randint(0,5) for j in range(4)]


# Angela Snyder's algorithm for MCMC on Mastermind
def mcmc2(code, initGuess, trials=100):
	av = None
	burnIn = 1000
	res = []
	for _ in range(trials):
		curGuess = initGuess[:]
		c = 0

		while curGuess != code:
			track = {}
			states = [initGuess]
			for _ in range(burnIn):
				prop = candCode(curGuess[:])
				curScore, candScore = compScore(curGuess, prop, code)
				if np.random.random() < candScore/curScore:
					curGuess = prop
					if tuple(prop) in track.keys():
						track[tuple(prop)] += 1
					else:
						track[tuple(prop)] = 1
				states.append(curGuess)
			curGuess = list(max(track.items(), key=operator.itemgetter(1))[0])

			c+=1
		res.append(c)
		if av is None:
			av = c
		else:
			av = (av + c)/2
	print("MCMC2 Method:", av)

	# Visualization of states and trials
	# states = np.array(states)
	# plt.hist(res, bins=50)
	# plt.show()
	# for i in range(4):
	# 	plt.plot(range(len(states)), states[:, i])
	# 	plt.plot(range(len(states)), np.full((len(states),), stats.mode(states[:, i])[0]))
	# 	plt.plot(range(len(states)), np.full((len(states),), np.mean(states[:, i])))
	# 	plt.show()



def mcmc(code, initGuess, trials=100):
	av = None
	res = []
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
		res.append(c)
	if av is None:
		av = c
	else:
		av = (av + c)/2
	plt.hist(res, 100)
	plt.show()
	print("MCMC Method:", av)
	# Visualization of the colors MCMC chooses to play for each slot
	states = np.array(states)
	for i in range(4):
		plt.plot(range(len(states)), states[:,i])
		plt.plot(range(len(states)), np.full((len(states),), stats.mode(states[:,i])[0]))
		plt.plot(range(len(states)), np.full((len(states),), np.mean(states[:, i])))
		plt.show()


def sample(cP):
	r = random.random()
	cSum = 0
	for i in range(6):
		cSum += cP[i]/sum(cP)
		if r <= cSum:
			return i
	return -1


def reinforcement(code, initGuess, trials=100):
	av = None
	alpha = .5
	res = []
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
				# curGuess = [random.randint(0,5) for q in Q]
			else:
				curGuess = [np.argmax(q) for q in Q]

			c += 1
			# print(np.argmax(np.array(Q), axis=1))
		res.append(c)
		if av is None:
			av = c
		else:
			av = (av + c) / 2
	print("Q Learning:", av)
	plt.hist(res, 50)
	plt.show()


code = [random.randint(0,5) for _ in range(4)]
print("Code:", code)

initGuess = [random.randint(0,5) for _ in range(4)]
print("Init Guess:", initGuess)

# pureRandom(code)
# randomNoReplace(code)
# genetic(code, initGuess)
# mcmc(code, initGuess)
# reinforcement(code, initGuess)
mcmc2(code, initGuess)

