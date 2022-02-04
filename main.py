import numpy as np
import pandas as pd
import random
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)


class myModel(Model):
    def __init__(self, n_agents, alpha, method, lex=[], correctNums=[]):
        super().__init__()
        self.schedule = BaseScheduler(self)
        for i in range(n_agents):
            a = MyAgent(i, self, alpha, method, lex, correctNums)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            agent_reporters={"revDist": "revDist",
                             "weakNonRevDist": "weakNonRevDist",
                             "strongNonRevDist": "strongNonRevDist",
                             "event": "event_type"
                             })

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


class MyAgent(Agent):
    def __init__(self, name, model, alpha, method, lex=[], correctNums=[]):
        super().__init__(name, model)
        self.name = name
        self.alpha = alpha
        self.method = method

        # Distribution is (SVO, SOV)

        # Solodata distributions
        self.revDist = (0.99, 0.01)
        self.weakNonRevDist = (0.58, 0.42)
        self.strongNonRevDist = (0.34, 0.66)

        # Correctness for (SVO, SOV)

        # Correctness from communication experiment
        # Only 2 datapoints for Reversible events with SOV type
        # self.correctRev = (0.98, 1)
        # self.correctWeakNon = (0.96, 0.95)
        # self.correctStrongNon = (0.96, 0.96)

        # Own numbers
        if correctNums == []:
            self.correctRev = (0.98, 0.1)
            self.correctWeakNon = (0.96, 0.3)
            self.correctStrongNon = (0.96, 0.96)
        else:
            self.correctRev = correctNums[0]
            self.correctWeakNon = correctNums[1]
            self.correctStrongNon = correctNums[2]

        # Lexical marker distribution
        if lex == []:
            self.lexicalRev = (0.28, 1)
            self.lexicalWeak = (0.014, 0.05)
            self.lexicalStrong = (0, 0.006)

        else:
            self.lexicalRev = lex[0]
            self.lexicalWeak = lex[1]
            self.lexicalStrong = lex[2]

        self.event_type = 0

    def step(self):
        # Select which agent to talk to
        other_agent = self.random.choice(self.model.schedule.agents)

        while other_agent.name == self.name:
            other_agent = self.random.choice(self.model.schedule.agents)

        # Event types are [Reversible, WeaklyNon, StronglyNon]
        self.event_type = random.choice([1, 2, 3])

        if self.method == "naive":
            if self.event_type == 1:
                svo, sov = self.revDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.revDist = dist

            elif self.event_type == 2:
                svo, sov = self.weakNonRevDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.weakNonRevDist = dist

            elif self.event_type == 3:
                svo, sov = self.strongNonRevDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.strongNonRevDist = dist

        elif self.method == "noisy_channel_single":
            if self.event_type == 1:
                svo, sov = self.revDist
                svoCor, sovCor = self.correctRev
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.revDist = dist

            elif self.event_type == 2:
                svo, sov = self.weakNonRevDist
                svoCor, sovCor = self.correctWeakNon
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.weakNonRevDist = dist

            else:
                svo, sov = self.strongNonRevDist
                svoCor, sovCor = self.correctStrongNon
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.strongNonRevDist = dist

        elif self.method == "Noisy Channel":
            if self.event_type == 1:
                svo, sov = self.revDist
                svoCor, sovCor = self.correctRev
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.revDist = dist

                svo2, sov2 = other_agent.revDist
                svoCor2, sovCor2 = other_agent.correctRev
                dist2 = other_agent.learn_noisy(svo2, sov2, svoCor2, sovCor2, other_agent.alpha)
                other_agent.revDist = dist2

            elif self.event_type == 2:
                svo, sov = self.weakNonRevDist
                svoCor, sovCor = self.correctWeakNon
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.weakNonRevDist = dist

                svo2, sov2 = other_agent.weakNonRevDist
                svoCor2, sovCor2 = other_agent.correctWeakNon
                dist2 = other_agent.learn_noisy(svo2, sov2, svoCor2, sovCor2, other_agent.alpha)
                other_agent.weakNonRevDist = dist2

            elif self.event_type == 3:
                svo, sov = self.strongNonRevDist
                svoCor, sovCor = self.correctStrongNon
                dist = self.learn_noisy(svo, sov, svoCor, sovCor, self.alpha)
                self.strongNonRevDist = dist

                svo2, sov2 = other_agent.strongNonRevDist
                svoCor2, sovCor2 = other_agent.correctStrongNon
                dist2 = other_agent.learn_noisy(svo2, sov2, svoCor2, sovCor2, other_agent.alpha)
                other_agent.strongNonRevDist = dist2

        elif self.method == "Static SVO Preference":
            if self.event_type == 1:
                svo, sov = self.revDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.revDist = dist
                self.incrementSVO()

            elif self.event_type == 2:
                svo, sov = self.weakNonRevDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.weakNonRevDist = dist
                self.incrementSVO()

            elif self.event_type == 3:
                svo, sov = self.strongNonRevDist
                dist = self.learn_naive(svo, sov, self.alpha)
                self.strongNonRevDist = dist
                self.incrementSVO()

        elif self.method == "Lexical Marker":
            if self.event_type == 1:
                svo, sov = self.revDist
                svoLex, sovLex = self.lexicalRev
                svoCor, sovCor = self.correctRev
                dist = self.learn_lexical(svo, sov, svoCor, sovCor, svoLex, sovLex, self.alpha)
                self.revDist = dist

            elif self.event_type == 2:
                svo, sov = self.weakNonRevDist
                svoLex, sovLex = self.lexicalWeak
                svoCor, sovCor = self.correctWeakNon
                dist = self.learn_lexical(svo, sov, svoCor, sovCor, svoLex, sovLex, self.alpha)
                self.weakNonRevDist = dist

            elif self.event_type == 3:
                svo, sov = self.strongNonRevDist
                svoLex, sovLex = self.lexicalStrong
                svoCor, sovCor = self.correctStrongNon
                dist = self.learn_lexical(svo, sov, svoCor, sovCor, svoLex, sovLex, self.alpha)
                self.strongNonRevDist = dist

    def learn_naive(self, svo, sov, alpha):
        x = random.random()

        if x <= svo:
            v = (1 - svo) * alpha
            svo += v
            sov -= v
        else:
            v = (1 - sov) * alpha
            svo -= v
            sov += v

        return (svo, sov)

    def learn_noisy(self, svo, sov, corSVO, corSOV, alpha):
        x = random.random()
        y = random.random()

        if x <= svo:
            if y <= corSVO:
                v = (1 - svo) * alpha
                svo += v
                sov -= v
        else:
            if y <= corSOV:
                v = (1 - sov) * alpha
                svo -= v
                sov += v

        return (svo, sov)

    def learn_lexical(self, svo, sov, corSVO, corSOV, svoLex, sovLex, alpha):
        x = random.random()
        y = random.random()
        z = random.random()

        if x <= svo:
            if z <= svoLex:
                corSVO = 1
            if y <= corSVO:
                v = (1 - svo) * alpha
                svo += v
                sov -= v
        else:
            if z <= sovLex:
                corSOV = 1
            if y <= corSOV:
                v = (1 - sov) * alpha
                svo -= v
                sov += v

        return (svo, sov)

    def incrementSVO(self):
        revSVO, revSOV = self.revDist
        weakSVO, weakSOV = self.weakNonRevDist
        strongSVO, strongSOV = self.strongNonRevDist

        revSVO += 0.001
        weakSVO += 0.001
        strongSVO += 0.001

        if revSVO > 1:
            revSVO = 1
        if weakSVO > 1:
            weakSVO = 1
        if strongSVO > 1:
            strongSVO = 1

        revSOV -= 0.001
        weakSOV -= 0.001
        strongSOV -= 0.001

        if revSOV < 0:
            revSOV = 0
        if weakSOV < 0:
            weakSOV = 0
        if strongSOV < 0:
            strongSOV = 0

        self.revDist = (revSVO, revSOV)
        self.weakNonRevDist = (weakSVO, weakSOV)
        self.strongNonRevDist = (strongSVO, strongSOV)


def visualize(steps, df, method):
    revs = np.zeros((steps, 1))
    weaks = np.zeros((steps, 1))
    strongs = np.zeros((steps, 1))

    for i in range(n_agents):
        agent = data.xs(i, level="AgentID")

        rev = [j for _, j in agent['revDist']]
        weak = [j for _, j in agent['weakNonRevDist']]
        strong = [j for _, j in agent['strongNonRevDist']]

        rev = np.reshape(rev, (steps, 1))
        weak = np.reshape(weak, (steps, 1))
        strong = np.reshape(strong, (steps, 1))

        revs = np.append(revs, rev, axis=1)
        weaks = np.append(weaks, weak, axis=1)
        strongs = np.append(strongs, strong, axis=1)

    revs = revs[:, 1:]
    weaks = weaks[:, 1:]
    strongs = strongs[:, 1:]

    avgRev = np.mean(revs, axis=1)
    avgWeak = np.mean(weaks, axis=1)
    avgStrong = np.mean(strongs, axis=1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    fig.suptitle(f"Average SOV choice per event type ({method})")

    ax1.plot(range(len(avgStrong)), avgStrong, label="Strongly Non")
    ax1.set_title("Strongly non-reversible events")
    ax1.set(xlabel='trials', ylabel='Proportion SOV')

    ax2.plot(range(len(avgWeak)), avgWeak, label="Weakly Non")
    ax2.set_title("Weakly non-reversible events")
    ax2.set(xlabel='trials')

    ax3.plot(range(len(avgRev)), avgRev, label="Reversible")
    ax3.set_title("Reversible events")
    ax3.set(xlabel='trials')


def combined_visual(steps, data, method):
    i = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    fig.suptitle(f"Average proportional SOV usage per event type over time ({method})")

    ax1.set_title("Strongly non-reversible events")
    ax1.set(xlabel='trials', ylabel='Proportion SOV')

    ax2.set_title("Weakly non-reversible events")
    ax2.set(xlabel='trials')

    ax3.set_title("Reversible events")
    ax3.set(xlabel='trials')

    for key, df in data.items():
        revs = np.zeros((steps, 1))
        weaks = np.zeros((steps, 1))
        strongs = np.zeros((steps, 1))

        for i in range(n_agents):
            agent = df.xs(i, level="AgentID")

            rev = [j for _, j in agent['revDist']]
            weak = [j for _, j in agent['weakNonRevDist']]
            strong = [j for _, j in agent['strongNonRevDist']]

            rev = np.reshape(rev, (steps, 1))
            weak = np.reshape(weak, (steps, 1))
            strong = np.reshape(strong, (steps, 1))

            revs = np.append(revs, rev, axis=1)
            weaks = np.append(weaks, weak, axis=1)
            strongs = np.append(strongs, strong, axis=1)

        revs = revs[:, 1:]
        weaks = weaks[:, 1:]
        strongs = strongs[:, 1:]

        avgRev = np.mean(revs, axis=1)
        avgWeak = np.mean(weaks, axis=1)
        avgStrong = np.mean(strongs, axis=1)

        ax1.plot(range(len(avgStrong)), avgStrong)
        ax2.plot(range(len(avgWeak)), avgWeak)
        ax3.plot(range(len(avgRev)), avgRev)

    ax3.legend(labels=["Distribution 1", "Distribution 2", "Distribution 3", "Distribution 4"])
    plt.show()


n_agents = 50
steps = 1000
methods = ["Lexical Marker"]

# Different values for correctness
nums = [
       [(0.96, 0.7), (0.96, 0.8), (0.96, 0.96)],
       [(0.96, 0.5), (0.96, 0.7), (0.96, 0.96)],
       [(0.96, 0.2), (0.96, 0.5), (0.96, 0.96)],
       [(0.96, 0.1), (0.96, 0.3), (0.96, 0.96)]]

# Different values for lexical distribution
# nums = [
#        [(0.28, 1), (0.014, 0.05), (0, 0.006)]]


for method in methods:
    multidata = {}
    j = 0
    for num in nums:
        model = myModel(n_agents, 0.03, method, correctNums=num)
        for i in range(steps):
            model.step()

        data = model.datacollector.get_agent_vars_dataframe()
        multidata[j] = data
        j += 1

    combined_visual(steps, multidata, method)
