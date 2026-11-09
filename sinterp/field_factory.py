import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any, Tuple, Union

from sinterp.stochasticfield import (
    TwoSidedField_v_etaz,
    TwoSidedField_b_s,
    TwoSidedField_x0_x1,
    TwoSidedField_b_eta1,
    OneSidedField_b,
    OneSidedField_eta1
)

def field_factory(
    field_name: str, 
    net_constructor: Callable[[int], Any], 
    interpolator: Any
) -> Any:
    """
    Creates a stochastic field model based on the field name.
    
    Args:
        field_name: Name of the field class to instantiate.
        net_constructor: A function that takes n_outputs and returns (net, *get_outputs).
        interpolator: The interpolator to use for the field.
        
    Returns:
        An instance of a StochasticField.
    """
    
    if field_name == "OneSidedField_eta1":
        net = net_constructor(n_outputs=1)
        return OneSidedField_eta1(interpolator, net)
        
    elif field_name == "OneSidedField_b":
        net = net_constructor(n_outputs=1)
        return OneSidedField_b(interpolator, net)
        
    elif field_name == "TwoSidedField_b_eta1":
        # Needs 2 outputs: b and eta1
        net, get_b, get_eta1 = net_constructor(n_outputs=2)
        b = lambda x, t, *args: get_b(net(x, t, *args))
        eta1 = lambda x, t, *args: get_eta1(net(x, t, *args))
        return TwoSidedField_b_eta1(interpolator, b, eta1)
        
    elif field_name == "TwoSidedField_v_etaz":
        # Needs 2 outputs: v and etaz
        net, get_v, get_etaz = net_constructor(n_outputs=2)
        v = lambda x, t, *args: get_v(net(x, t, *args))
        etaz = lambda x, t, *args: get_etaz(net(x, t, *args))
        return TwoSidedField_v_etaz(interpolator, v, etaz)
        
    elif field_name == "TwoSidedField_b_s":
        # Needs 2 outputs: b and s
        net, get_b, get_s = net_constructor(n_outputs=2)
        b = lambda x, t, *args: get_b(net(x, t, *args))
        s = lambda x, t, *args: get_s(net(x, t, *args))
        return TwoSidedField_b_s(interpolator, b, s)
        
    elif field_name == "TwoSidedField_x0_x1":
        # Needs 2 outputs: eta0 and eta1
        net, get_eta0, get_eta1 = net_constructor(n_outputs=2)
        eta0 = lambda x, t, *args: get_eta0(net(x, t, *args))
        eta1 = lambda x, t, *args: get_eta1(net(x, t, *args))
        return TwoSidedField_x0_x1(interpolator, eta0, eta1)
        
    else:
        raise ValueError(f"Unknown field_name: {field_name}")

def make_default_net_constructor(
    init_net: Callable[[int], eqx.Module]
) -> Callable[[int], Union[eqx.Module, Tuple[eqx.Module, Callable, Callable]]]:
    """
    Creates a net_constructor that uses a provided init_net function.
    Supports PyTree/dict inputs and outputs via jax.tree_util.
    
    Args:
        init_net: A function that takes n_outputs and returns an eqx.Module.
        
    Returns:
        A net_constructor function.
    """
    def net_constructor(n_outputs=1):
        # We pass n_outputs to init_net and let it handle dimension logic.
        net = init_net(n_outputs)
        
        if n_outputs == 1:
            return net
        
        if n_outputs == 2:
            # Use tree_map to split the output PyTree across the last dimension
            get_1st_out = lambda x: jax.tree_util.tree_map(
                lambda leaf: leaf[..., :leaf.shape[-1]//2], x
            )
            get_2nd_out = lambda x: jax.tree_util.tree_map(
                lambda leaf: leaf[..., leaf.shape[-1]//2:], x
            )
            return net, get_1st_out, get_2nd_out
            
        raise ValueError(f"Unsupported n_outputs: {n_outputs}")
        
    return net_constructor
