import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple


@dataclass
class NetworkParameters:
    # input -> hidden
    w1: float = 0.2
    w2: float = -0.3
    b1: float = 0.4

    w3: float = -0.5
    w4: float = 0.1
    b2: float = -0.2

    # hidden -> output
    w5: float = 0.3
    w6: float = -0.4
    b3: float = 0.2


@dataclass
class ForwardCache:
    x1: float
    x2: float
    z1: float
    h1: float
    z2: float
    h2: float
    z3: float
    y_hat: float


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative_from_output(sigmoid_output: float) -> float:
    return sigmoid_output * (1.0 - sigmoid_output)


def forward_propagation(x1: float, x2: float, params: NetworkParameters) -> ForwardCache:
    # Hidden neuron h1
    z1 = params.w1 * x1 + params.w2 * x2 + params.b1
    h1 = sigmoid(z1)

    # Hidden neuron h2
    z2 = params.w3 * x1 + params.w4 * x2 + params.b2
    h2 = sigmoid(z2)

    # Linear output neuron
    z3 = params.w5 * h1 + params.w6 * h2 + params.b3
    y_hat = z3

    return ForwardCache(
        x1=x1,
        x2=x2,
        z1=z1,
        h1=h1,
        z2=z2,
        h2=h2,
        z3=z3,
        y_hat=y_hat,
    )


def mean_squared_error(y_hat: float, y_true: float) -> float:
    return 0.5 * (y_hat - y_true) ** 2


def backpropagation(cache: ForwardCache, y_true: float, params: NetworkParameters) -> Dict[str, float]:
    # Output layer error (linear neuron)
    delta3 = cache.y_hat - y_true

    # Gradients for output layer
    dw5 = delta3 * cache.h1
    dw6 = delta3 * cache.h2
    db3 = delta3

    # Hidden layer errors
    delta1 = delta3 * params.w5 * sigmoid_derivative_from_output(cache.h1)
    delta2 = delta3 * params.w6 * sigmoid_derivative_from_output(cache.h2)

    # Gradients for hidden layer
    dw1 = delta1 * cache.x1
    dw2 = delta1 * cache.x2
    db1 = delta1

    dw3 = delta2 * cache.x1
    dw4 = delta2 * cache.x2
    db2 = delta2

    return {
        "delta3": delta3,
        "delta1": delta1,
        "delta2": delta2,
        "dw1": dw1,
        "dw2": dw2,
        "db1": db1,
        "dw3": dw3,
        "dw4": dw4,
        "db2": db2,
        "dw5": dw5,
        "dw6": dw6,
        "db3": db3,
    }


def gradient_descent_step(params: NetworkParameters, grads: Dict[str, float], eta: float = 0.1) -> NetworkParameters:
    return NetworkParameters(
        w1=params.w1 - eta * grads["dw1"],
        w2=params.w2 - eta * grads["dw2"],
        b1=params.b1 - eta * grads["db1"],
        w3=params.w3 - eta * grads["dw3"],
        w4=params.w4 - eta * grads["dw4"],
        b2=params.b2 - eta * grads["db2"],
        w5=params.w5 - eta * grads["dw5"],
        w6=params.w6 - eta * grads["dw6"],
        b3=params.b3 - eta * grads["db3"],
    )


def print_report():
    x1, x2 = 0.6, 0.1
    y_true = 0.8
    eta = 0.1

    params = NetworkParameters()
    cache = forward_propagation(x1, x2, params)
    loss = mean_squared_error(cache.y_hat, y_true)
    grads = backpropagation(cache, y_true, params)
    updated_params = gradient_descent_step(params, grads, eta)

    print("=== FORWARD PROPAGATION ===")
    print(f"z1   = {cache.z1:.6f}")
    print(f"h1   = {cache.h1:.6f}")
    print(f"z2   = {cache.z2:.6f}")
    print(f"h2   = {cache.h2:.6f}")
    print(f"y_hat= {cache.y_hat:.6f}")
    print(f"MSE  = {loss:.6f}")

    print("\n=== BACKPROPAGATION ===")
    for key in ["delta3", "delta1", "delta2", "dw1", "dw2", "db1", "dw3", "dw4", "db2", "dw5", "dw6", "db3"]:
        print(f"{key:>6} = {grads[key]:.6f}")

    print("\n=== UPDATED WEIGHTS (eta = 0.1) ===")
    for key, value in asdict(updated_params).items():
        print(f"{key} = {value:.6f}")


if __name__ == "__main__":
    print_report()
