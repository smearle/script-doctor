import jax.numpy as jnp
import jax


# Function to stack leaves of PyTrees
def stack_leaves(trees):
    # Make sure each leaf is an array
    def to_array(x):
        x = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
        # if x.shape == ():
        #     x = x.reshape(1)
        return x

    trees = [jax.tree.map(lambda x: to_array(x), tree) for tree in trees]

    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in trees]
    flat_trees, treedefs = zip(*flat_trees_treedefs)

 
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.stack(leaves) for leaves in zip(*flat_trees)]

    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)

