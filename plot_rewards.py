import json
import matplotlib.pyplot as plt
import os
import numpy as np

no_clusters = os.listdir("sac_model/no_cluster")
clusters = os.listdir("sac_model/cluster_2")

no_cluster_rewards = []
for i in range(5):
    f = open(os.path.join("sac_model","no_cluster", no_clusters[i], "rewards"))
    rewards = json.load(f)
    no_cluster_rewards.append(rewards['Rewards'])
    f.close()

cluster_rewards = []
for i in range(5):
    f = open(os.path.join("sac_model","cluster_2", clusters[i], "rewards"))
    rewards = json.load(f)
    cluster_rewards.append(rewards['Rewards'][-250:])
    f.close()

for i in range(5):
    plt1 = plt.plot(np.array(cluster_rewards[i], dtype=float), color="blue")
    plt2 = plt.plot(np.array(no_cluster_rewards[i], dtype=float), color="orange")

plt.show()
