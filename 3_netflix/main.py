import numpy as np
#import kmeans
import common
#import naive_em
import em

#X = np.loadtxt("toy_data.txt")

X = np.loadtxt("football_incomplete.txt")
K = 4
seed = np.array([0,1,2,3,4])
max = -9999
for i in seed:    
    mixture, post = common.init(X, K, seed[i])
    mixture, post, log_likelihood = em.run(X, mixture, post) 
    if(log_likelihood > max):
        max = log_likelihood
        best_mixture = mixture
print("max ", max)
X_pred = em.fill_matrix(X, best_mixture)
print(X_pred)

#Problem 8-1: report log likelihood on Netflix data
# X = np.loadtxt("netflix_incomplete.txt")
# K = 12
# seed = np.array([0,1,2,3,4])
# max = -2521060
# for i in seed:    
#     mixture, post = common.init(X, K, seed[i])
#     mixture, post, log_likelihood = em.run(X, mixture, post) 
#     if(log_likelihood > max):
#         max = log_likelihood
#         best_mixture = mixture
# print("max ", max)
 
# Problem 8-2: compare with gold 
# X_gold = np.loadtxt('netflix_complete.txt')
#  
# X_pred = em.fill_matrix(X, best_mixture)
# print(X_pred)
# print(common.rmse(X_gold, X_pred))


# Problem 4-2: find the best K and best BIC       
# K = 4
# seed = np.array([0,1,2,3,4])
# max = -1500
# for i in seed:    
#     mixture, post = common.init(X, K, seed[i])
#     mixture, post, log_likelihood = naive_em.run(X, mixture, post) 
#     if(log_likelihood > max):
#         max = log_likelihood
#         best_mixture = mixture
# print("max ", max)
# print(best_mixture)
#score = common.bic(X, best_mixture, max)
# print(score)

# Problem 3-2: analyze k-mean and em for each K
# K = 4
# seed = 0
# mixture, post = common.init(X, K, seed)
# mixture, post, cost = kmeans.run(X, mixture, post)
# title = "Kmeans: K=" + str(K)
# common.plot(X, mixture, post, title)
# 
# mixture, post = common.init(X, K, seed)
# mixture, post, cost = naive_em.run(X, mixture, post)
# title = "EM: K=" + str(K)
# common.plot(X, mixture, post, title)

# Problem 3-1: find highest log-likelihood
# K = 4
# seed = np.array([0,1,2,3,4])
# max = -1138.891045633598
# for i in seed:    
#     mixture, post = common.init(X, K, seed[i])
#     mixture, post, log_likelihood = naive_em.run(X, mixture, post) 
    #print("current ", log_likelihood)
#     if(log_likelihood > max):
#         max = log_likelihood
#         best_mixture = mixture
# print("max ", max)

# Problem 2: implement naive_em.py 
# K = 3
# seed = 0
# mixture, post = common.init(X, K, seed)
# mixture, post, cost = naive_em.run(X, mixture, post)
# print(cost)

# Problem 1 and 3: find minimum cost
# K = 1
# seed = np.array([0,1,2,3,4])
# min_cost = 99999
# for i in seed:    
#     mixture, post = common.init(X, K, seed[i])
#     mixture, post, cost = kmeans.run(X, mixture, post) 
#     if cost < min_cost:
#         min_cost = cost
#  
# print(min_cost)
        