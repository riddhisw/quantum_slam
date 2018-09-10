'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: control_action

    :synopsis: Implements controller specifying measurement locations for qslam.

    Module Level Functions:
    ----------------------
        control_lengthscale_uncertainty : Return location(s) for next set of
            single qubit measurement(s) selecting qubit locations with highest state
            estimate uncertainty.

        control_user_input : Return location(s) for next set oF single qubit
            measurement(s) as defined by the user.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

import numpy as np

def control_lengthscale_uncertainty(listofcontrolparameters,
                                    next_control_neighbourhood,
                                    number_of_diff_nodes=1,
                                    dtype=[('Node', int), ('ControlParam', float)]
                                   ):

    ''' Return location(s) for next set of single qubit measurement(s)
        selecting qubit locations with highest state estimate uncertainty.

    Parameters:
    ----------
        listofcontrolparameters (`float64`| numpy array):
            Avg. uncertainity metric for correlation lengthscales at each node.
            Dims: Grid.number_of_nodes

        next_control_neighbourhood (`int`| list) :
            List of indices for a qubit within a control region.

        number_of_diff_nodes (`int`| scalar | optional):
            Number of single qubit measurements at different locations
            that can be simultaneously performed on the hardware grid.

        dtype ( List of tuples | optional) :
            Specifies how control parameters and control neighbourhoods are read,
            and handled by this function. Should be a hidden local variable.

    Returns:
    -------
        controls_list (`float64` | numpy array):
            Location(s) for performnng the next single qubit measurement.
            Dims: number_of_diff_nodes
    '''

    # COMMENT: store control parameters in a structureed numpy array.
    # Then, mask control parameters which are corresspond to be out of the
    # control neighbourhood. If the control is empty, include all qubits
    # in the analysis for choosing the next measurement.

    labelled_params = [iterate for iterate in enumerate(listofcontrolparameters)]
    structured_array = np.asarray(labelled_params, dtype=dtype)

    if len(next_control_neighbourhood) == 0:
        mask = np.ones(len(labelled_params), dtype=bool) # Mask for showing all values.
        print "Control List empty; randomly chosen qubit on grid."

    if len(next_control_neighbourhood) > 0:
        mask = np.zeros(len(labelled_params), dtype=bool) # Mask for hiding all values.
        mask[next_control_neighbourhood] = True # Show nodes only from next_control_neighbourhood.

    structured_array = structured_array[mask] # No mask for lists - make array and bring back.

    # Now, sort the region from max to min uncertainty.
    sorted_array = np.sort(structured_array, order=['ControlParam', 'Node'])[::-1]

    # Zeroth term of sorted_array is the node corresponding to maximal ucertainity.
    max_val = sorted_array['ControlParam'][0]

    # Now find multiple instances of the maximum value if they exist.
    # If the counter > 1, then multiple maxima exist.
    counter = 1
    for controlparamval in sorted_array['ControlParam'][1:]:
        if controlparamval == max_val:
            counter += 1
        elif controlparamval != max_val:
            break

    # Return controls based on highest uncertainty.
    # If equi-certain options exist, choose between these options at random.
    if number_of_diff_nodes < counter:
        # Returns random choices among equicertain nodes.
        chosen_node_indices = np.random.randint(low=0, high=counter, size=number_of_diff_nodes)
        return sorted_array['Node'][chosen_node_indices]

    elif number_of_diff_nodes >= counter:
        # Return nodes in descending order of uncertainty
        return sorted_array['Node'][0 : number_of_diff_nodes]


def control_user_input(input_controls, next_control_neighbourhood):
    ''' Return location(s) for next set of single qubit measurement(s). No control
    protocol specified. '''

    # do nothing

    return input_controls


PROTOCOL = {"userinput" : control_user_input,
            "lenvar": control_lengthscale_uncertainty
           }


def controller(listofcontrolparameters, next_control_neighbourhood, controltype='lenvar', number_of_nodes=1):
    ''' Return location(s) for next set of single qubit measurement(s) based on
        a selected control protocol.

        Parameters:
        ----------
        listofcontrolparameters (`float64`| numpy array-like):
            List of input information for the control protocol specified by controltype.

        controltype (`str`| optional):
            Specifies control protocol to be used:
                'lenvar' : Calls protocol control_lengthscale_uncertainty.
                'userinput' : Returns listofcontrolparameters (no control).
            Defaults to 'lenvar'.

        number_of_nodes (`int`| scalar | optional):
            Number of single qubit measurements that can be simultaneously
            performend on the hardware grid.
            Defaults to 1.

        Returns:
        -------
        controls_list (`float64` | numpy array):
            Location(s) for performnng the next single qubit measurement.
            Dims: number_of_nodes
    '''

    if len(listofcontrolparameters) > 0:

        controls_list = PROTOCOL[controltype](listofcontrolparameters, next_control_neighbourhood, number_of_nodes)
        return controls_list[0:number_of_nodes]

    elif len(listofcontrolparameters) == 0:

        print "Input control parameters empty."
        print "Runtime Error raised"
        raise RuntimeError
