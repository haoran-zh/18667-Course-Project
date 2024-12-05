#!/bin/bash
# client fraction is all 0.1 (set in each train function)
# can change client number
iidalpha=0.1 # lower value means more non-iid (more challenging)
NumClients=50

python plots.py --samplingAlgo bandit_sampling --num-clients $NumClients --iid-alpha $iidalpha
python plots.py --samplingAlgo full_participation_train --num-clients $NumClients --iid-alpha $iidalpha
python plots.py --samplingAlgo random_sampling_train --num-clients $NumClients --iid-alpha $iidalpha # uniform distribution
python plots.py --samplingAlgo dataset_size_sampling_train --num-clients $NumClients --iid-alpha $iidalpha
python plots.py --samplingAlgo variance_reduced_sampling_train --num-clients $NumClients --iid-alpha $iidalpha
python plots.py --samplingAlgo poc_sampling --num-clients $NumClients --iid-alpha $iidalpha
python plots.py --samplingAlgo random_sampling --num-clients $NumClients --iid-alpha $iidalpha # random, fixed active client number