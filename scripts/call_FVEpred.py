import networkx as nx
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from topoexpsearch_FVE.srv import pred_FVE

from std_msgs.msg import String
import ast
import numpy as np
import copy

NN = nx.Graph()
NN.add_nodes_from([(0, {'pos': (-28., 0.), 'type': 4, 'is_robot': True, 'to_go': False, 'value': 0}),
                   (1, {'pos': (-35., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (2, {'pos': (-28., 5.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.9}),
                   (3, {'pos': (-28., -5.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (4, {'pos': (-21., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1})])
NN.add_edges_from([(0, 1), (0,2), (0, 3), (0, 4)])
# convert NN to dict
dict_G = {}
E = [[v1,v2] for (v1,v2) in list(nx.edges(NN))]
C = {str(n):str(c) for n, c in nx.get_node_attributes(NN, 'type').items()}
dict_G["edges"] = E
dict_G["features"] = C

msg_NN_jsonstr = String()
msg_NN_jsonstr.data = str(dict_G)

v = 0

# p_E : [1xN_c] probability vector
p_E = [0.5, 0.1, 0.1, 0.1, 0.1]
msg_p_E = String()
msg_p_E.data = str(p_E)

# rollout length
K = 3

# high member cut
M = 2

# number of class
N_c = 5

# discount factor
gamma = 1

rospy.wait_for_service('pred_FVE')
srv_pred_MBM = rospy.ServiceProxy('pred_FVE', pred_FVE)
output = srv_pred_MBM(msg_NN_jsonstr, v, msg_p_E, N_c, M, K, gamma)
print(output.p_o)