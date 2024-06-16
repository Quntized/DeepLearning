def cost_calculate(y_true,y_prediction):
    return np.sum((y_true - y_prediction)**2)/len(y_true)

def activation_function(y_prediction):
    return (1/(1+np.exp(y_prediction)))

def Gradient_descent(X,Y,iteration,stopping_threshold):
    costs = []
    learning_rate = 0.0001
    store_weight = []
    weights = np.random.randn(4,3)
    bias = np.random.rand(4,1)
    for i in range(iteration):
        y_prediction = np.dot(weights,X) + bias
        y_after_activation = activation_function(y_prediction)
        current_cost = cost_calculate(Y,y_after_activation)
        costs.append(current_cost)
        store_weight.append(weights)
        if previous_cost and np.abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
        weight_derivative = (-2/n)*np.sum(X * (Y-(np.dot(weight,X)+bias)))
        bias_derivative = (-2/n)*np.sum((Y - (np.dot(weight,X)+bias)))
        weights = weights - (learning_rate * weight_derivative)
        bias = bias - (learning_rate * bias_derivative)
        print(f"Iteration {i+1}: Cost {current_cost}, Weight {weights}, Bias {bias}")
    
    
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
     
    return weights, bias


def main():
    X = np.array([1,2,3])
    Y= np.array([2,3,4,5])
    weight , bias = Gradient_descent(X,Y,iterations=100,stopping_threshold = 1e-6)
    print(weight,bias)