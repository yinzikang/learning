import torch
import numpy as np

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function that takes in PyTorch Tensors for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
PyTorch Tensor for (previously-generated) samples from those 
distributions, and returns a Tensor containing the log 
likelihoods of those samples.

"""


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: action, Tensor with shape [batch, dim]
        mu: means, Tensor with shape [batch, dim]
        log_std: log stds, Tensor with shape [batch, dim] or [dim]

    Returns:
        log likelihoods, Tensor with shape [batch]
    """

    # log_likelihoods = - 1 / 2 * torch.sum(torch.div(torch.pow(x - mu, 2), torch.pow(torch.exp(log_std), 2))
    #                                        + 2 * log_std + np.log(2*np.pi), axis=-1)
    # log_likelihoods = -0.5 * torch.sum(torch.pow(torch.div(x - mu, torch.exp(log_std)), 2)
    #                                    + 2 * log_std + np.log(2 * np.pi), axis=-1)
    log_likelihoods = -0.5 * torch.sum(((x - mu)/torch.exp(log_std)) ** 2
                                       + 2 * log_std + np.log(2 * np.pi), axis=-1)
    # log_likelihoods = -0.5 * ((x - mu) / torch.exp(log_std)) ** 2 + 2 * log_std + np.log(2 * np.pi)

    return log_likelihoods
    # return log_likelihoods.sum(axis=-1)


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.pytorch.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    batch_size = 32
    dim = 10

    x = torch.rand(batch_size, dim)
    mu = torch.rand(batch_size, dim)
    log_std = torch.rand(dim)

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

    your_result = your_gaussian_likelihood.detach().numpy()
    true_result = true_gaussian_likelihood.detach().numpy()

    correct = np.allclose(your_result, true_result)
    print_result(correct)
