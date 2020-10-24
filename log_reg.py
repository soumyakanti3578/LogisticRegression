import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def read_and_process(filepath, delim=","):
    data = pd.read_csv(filepath, delimiter=delim)
    data = (data - data.min())/(data.max()-data.min())
    data.loc[data["label"] == 0, "label"] = -1
    data["bias"] = 1
    
    z = data[["bias", "exam1", "exam2"]].multiply(data["label"], axis="index")
    z = z.values
    
    y = data["label"].values.reshape((z.shape[0], 1))

    return data, z, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def g3(x):
    return 0.5 - 1.20096*(x/8) + 0.81562*((x/8)**2)

def g7(x):
    return 0.5 - 1.73496*(x/8) + 4.19407*(x/8)**3 - 5.43402*(x/8)**5 + 2.50739*(x/8)**7

def cost(x):
    return np.log(1 + np.exp(-x)).mean()

def gradient_descent(Z, alpha=0.01, gamma=0.5, iters=100):
    m = Z.shape[0]
    v = np.ones((1, Z.shape[1]))
    betas = np.ones((1, Z.shape[1]))
    costs = []
    
    for _ in range(iters):
        g = sigmoid(-np.matmul(Z, v.T))
        betas_new = v + alpha * np.matmul(g.T, Z)
        v = (1-gamma)*betas_new + gamma*betas
        betas = betas_new
        
        costs.append(cost(np.matmul(Z, v.T)))
    
    return betas, costs
    
def plot_costs(costs):
	plt.plot([i for i in range(len(costs))], costs)
	plt.show()
    
def main():
	data, z, y = read_and_process("exams.csv")
	params, costs = gradient_descent(z, alpha=0.2, iters=50)
	print(params)
	plot_costs(costs)
	
if __name__ == "__main__":
	main()
