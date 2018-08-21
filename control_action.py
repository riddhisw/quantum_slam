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
                                    number_of_nodes,
                                    dtype = [('Node', int), ('ControlParam', float)]
                                    ):
# def control_lengthscale_uncertainty(listofcontrolparameters, number_of_nodes):
    ''' Return location(s) for next set of single qubit measurement(s)
        selecting qubit locations with highest state estimate uncertainty.

    Parameters:
    ----------
        listofcontrolparameters (`float64`| numpy array):
            Avg. uncertainity metric for correlation lengthscales at each node.
            Dims: Grid.number_of_nodes

        next_control_neighbourhood (`int`| list) :
            List of indices for a qubit within a control region.

        number_of_nodes (`int`| scalar):
            Number of single qubit measurements that can be simultaneously
            performed on the hardware grid.

    Returns:
    -------
        controls_list (`float64` | numpy array):
            Location(s) for performnng the next single qubit measurement.
            Dims: number_of_nodes
    '''

    labelled_params = [iterate for iterate in enumerate(listofcontrolparameters)]
    structured_array = np.asarray(labelled_params, dtype=dtype)
    # Take out those parameters not in the control region
    # Assume that the Qubit Grid is convex. Any convex polygon can be triangulated
    # Use this to justify, consider only three nearest neighbours in picking from neighbourhood. (No three point co-linear)
    # This argument may not apply but its a start

    if 3 <= len(next_control_neighbourhood):
        print "the neighbourhood has at least 3 qubits"
        if len(next_control_neighbourhood) < len(listofcontrolparameters) - 3:
            print "the neighbourhood is very large"
            mask = np.zeros(len(labelled_params), dtype=bool) # Mask all values
            mask[next_control_neighbourhood] = True # Show nodes only from next_control_neighbourhood
            structured_array = structured_array[mask]#No mask for lists - make array and bring back
            print "I truncated something, see that the control params are shorter", len(structured_array)

    # Now, sort the region from max to min uncertainty.
    # structured_array = np.asarray(labelled_params, dtype=dtype)
    sorted_array = np.sort(structured_array, order=['ControlParam', 'Node'])[::-1]

    # Find multiple instances of maximal uncertainty
    max_val = sorted_array['ControlParam'][0]
    print "max value is ", max_val, ", at node, ", sorted_array['Node'][0]
    counter = 1
    for controlparamval in sorted_array['ControlParam'][1:]:
        if controlparamval == max_val:
            counter += 1
        elif controlparamval != max_val:
            break

    # Return controls based on highest uncertainty.
    # If equi-certain options exist, choose between these options at random.
    number_of_nodes = 2
    if number_of_nodes < counter:
        # Returns random choices among equicertain nodes.
        chosen_node_indices = np.random.randint(low=0, high=counter, size=number_of_nodes)
        return sorted_array['Node'][chosen_node_indices]

    elif counter <= number_of_nodes:
        # Returns nodes in descending order of uncertainty.
        return sorted_array['Node'][0 : number_of_nodes]

    # controls_list = []
    # print "here is the input list of r-variances for control"
    # print listofcontrolparameters

    # for idxnode in range(number_of_nodes):

    #     # TODO: enumerate and sort?
    #     node_j = np.argmax(listofcontrolparameters)

    #     if  node_j > -1.0:
    #         # TODO: Eliminate node from future potential controls - why?
    #         listofcontrolparameters[node_j] = -1.0
    #         controls_list.append(node_j)

    #     elif node_j == -1.0:

    #         if len(controls_list) > 0:
    #             return controls_list

    #         elif len(controls_list) == 0:
    #             print "ERROR in control_lengthscale_uncertainty"
    #             print "List of controls  = ", controls_list
    #             raise RuntimeError

    # return np.asarray(controls_list)


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
        raise RuntimeError

