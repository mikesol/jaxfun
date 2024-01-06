def truncate_if_odd(tensor):
    """
    Truncate the tensor along the batch dimension if the batch size is odd.
    
    Args:
    tensor (jax.numpy.ndarray): A tensor of shape (batch, seq, features).

    Returns:
    jax.numpy.ndarray: The original tensor or the tensor truncated along the batch dimension.
    """
    batch_size = tensor.shape[0]
    if batch_size % 2 != 0:
        # Truncate the last element in the batch dimension if the batch size is odd
        return tensor[:-1]
    else:
        # Return the tensor as is if the batch size is even
        return tensor