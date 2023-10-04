# script for calculating posterior probability of uber having N cars given the highest observed number of cars.
# #Author: Petter Byström

from matplotlib import pyplot as plt
import math
N = range(60,1000)
Nmax=1000
def Pd_n(N):
    return 1/N 

def Pn(Nmax):
    return 1/Nmax

def Pd(D,Nmax):
    return sum([Pn(Nmax)*Pd_n(i) for i in range(D,Nmax)])

def Pn_d(D,Nmax,N):
    return Pd_n(N)*Pn(Nmax)/Pd(D,Nmax)


def En_d(D,Nmax):
    #return 1/sum([1/i for i in range(D,Nmax)])
    return sum([i*Pn_d(D,Nmax,i) for i in range(60,1000)])
    #return math.log10(Nmax/D)*sum([i*Pn(Nmax) for i in range(D,Nmax)])
def mean_prob(prob):
    return sum(prob)/len(prob)

print(En_d(60,1000))
print(sum([i*Pn_d(60,Nmax, i) for i in N]))
#plt.plot(N, [Pn_d(60,Nmax, i) for i in N])
""" plt.plot([1/i for i in N],N)
plt.ylabel('value of N')
plt.xlabel('Posterior probability')
plt.title('Posterior distribution')

#print(sum([Pn_d(60,Nmax, i) for i in N]))
plt.show() """

