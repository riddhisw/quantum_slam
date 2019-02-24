import numpy as np

DataSetProperties_1 = {}

DataSetProperties_1['key'] = 1
DataSetProperties_1['path'] =  '/home/riddhisw/Documents/SLAM_project/project/statedetectn/rf_fulldata_ramsey.npz'
DataSetProperties_1['wait_time'] = 40 # milliseconds
DataSetProperties_1['classifier'] = 'Random Forest with Importance Re-weighting'
DataSetProperties_1['expttype'] = 'Ramsey'
DataSetProperties_1['parameters'] = np.load(DataSetProperties_1['path'])['DataParams'].item()
DataSetProperties_1['parameters']['dpts'] = 1 # same experiment (static)

DataSetProperties_2 = {}

DataSetProperties_2['key'] = 2
DataSetProperties_2['path'] =  '/home/riddhisw/Documents/SLAM_project/project/statedetectn/mlp_fulldata_ramsey.npz'
DataSetProperties_2['wait_time'] = 40 # milliseconds
DataSetProperties_2['classifier'] = 'MultiLayer Perceptron'
DataSetProperties_2['expttype'] = 'Ramsey'
DataSetProperties_2['parameters'] = np.load(DataSetProperties_2['path'])['DataParams'].item()
DataSetProperties_2['parameters']['dpts'] = 1 # same experiment (static)

DataSetProperties_3 = {}

DataSetProperties_3['key'] = 3
DataSetProperties_3['path'] =  '/home/riddhisw/Documents/SLAM_project/project/statedetectn/rf_fulldata__pApB_8.npz'
DataSetProperties_3['wait_time'] = 8 # milliseconds
DataSetProperties_3['classifier'] = 'Random Forest with  Importance Re-weighting'
DataSetProperties_3['expttype'] = 'pA or pB'
DataSetProperties_3['parameters'] = np.load(DataSetProperties_3['path'])['DataParams'].item()
DataSetProperties_3['parameters']['dpts'] = 2 # two different types of expts - pA and pB

DataSetProperties_4 = {}

DataSetProperties_4['key'] = 4
DataSetProperties_4['path'] =  '/home/riddhisw/Documents/SLAM_project/project/statedetectn/rf_fulldata__pApB_8.npz'
DataSetProperties_4['wait_time'] = 25 # milliseconds
DataSetProperties_4['classifier'] = 'MultiLayer Perceptron'
DataSetProperties_4['expttype'] = 'pA or pB'
DataSetProperties_4['parameters'] = np.load(DataSetProperties_4['path'])['DataParams'].item()
DataSetProperties_4['parameters']['dpts'] = 2 # Two different types of expts - pA and pB

DataKeys = {'1': DataSetProperties_1,
            '2': DataSetProperties_2,
            '3': DataSetProperties_3,
            '4': DataSetProperties_4
}



class RealData(object):

    def __init__(self, data_key, choose_expt=0):
        ''' Accesses output classifier data of prob of seeing a bright ion by Hempel et al. '''
        
        
        self.DataProp = DataKeys[str(data_key)]
        self.ions = self.DataProp['parameters']['N']
        self.dpts = self.DataProp['parameters']['dpts'] 
        self.img_shape = self.DataProp['parameters']['img_shape']
        self.choose_expt = choose_expt
        
        
        data = np.load(self.DataProp['path']).files
        data.remove('DataParams')
        
        for idx_element_name in data:
            setattr(RealData, idx_element_name, np.load(self.DataProp['path'])[idx_element_name])
        
        self.expt_repetitions = int(self.binary_data.shape[1] / self.dpts)
        
        # Sampling without replacement at each node. Dummy helper variables.
        self.sample_repts = np.zeros((self.ions, self.expt_repetitions))
        self.sample_repts[:] = np.arange(self.expt_repetitions)
        
    

    def get_real_data(self, node_j):
        '''Return a msmt from analysis of an experimental dataset
        node_j: postion index for ion
        '''

        pick_rep = self.sample_repetitions_without_replacement(node_j)
        start = int(self.choose_expt * self.expt_repetitions)
        stop = int(start + self.expt_repetitions)
        image_data_point = self.binary_data[node_j, start :  stop][pick_rep]
        
        return image_data_point


    def sample_repetitions_without_replacement(self, node_j):
        ''' Samples a database of experimental measurements as if conducted
        in real time. Sampling occurs without replacement.'''

        total_samples_left = len(set(self.sample_repts[node_j, :])) # remove duplicates

        if total_samples_left > 1:
            
            pick_rep = -1
            while pick_rep < 0:
                idx = np.random.randint(low=0, high=total_samples_left)
                pick_rep = int(list(set(self.sample_repts[node_j, :]))[idx])
            
            self.sample_repts[node_j, pick_rep] = -1 # remove from next iteration

            return pick_rep

        elif total_samples_left == 1:
            print("No more expt measurements avail at node:", node_j)
            raise RuntimeError
