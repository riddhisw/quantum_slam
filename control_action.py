'''
MODULE: control_action

Supports variance based control on filtering.

'''

import numpy as np


def controller(listofcontrolparameters, number_of_nodes=1):
    '''docstring'''
    print
    print "listofcontrolparameters = ", listofcontrolparameters

    if len(listofcontrolparameters) > 0:

        controls_list = []

        for idxnode in range(number_of_nodes):
            node_j = np.argmax(listofcontrolparameters) # why not sort?

            if  node_j > -1.0:
                listofcontrolparameters[node_j] = -1.0 # eliminate node from analysis # # wait, why?
                controls_list.append(node_j)

            elif node_j == -1.0:

                if len(controls_list) > 0:
                    return controls_list

                print "controls_list empty. No valid control remaining."
                return controls_list

        print "controls_list = ", controls_list
        print
        return controls_list[0:number_of_nodes]

    elif len(listofcontrolparameters) == 0:

        print "Input control parameters empty."
        raise RuntimeError
