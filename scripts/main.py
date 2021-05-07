import networkx as nx
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from std_msgs.msg import String
import ast
import numpy as np
import copy

class Frontier_Value_Estimation():
    def __init__(self):
        rospy.wait_for_service('pred_MBM')
        self.srv_pred_MBM = rospy.ServiceProxy('pred_MBM', pred_MBM)

    def srv_MBM(self, G, v):
        # convert NN to dict
        dict_G = {}
        E = [[v1, v2] for (v1, v2) in list(nx.edges(G))]
        C = {str(n): str(c) for n, c in nx.get_node_attributes(G, 'type').items()}
        dict_G["edges"] = E
        dict_G["features"] = C

        msg_NN_jsonstr = String()
        msg_NN_jsonstr.data = str(dict_G)
        node = v

        output = self.srv_pred_MBM(msg_NN_jsonstr, node)
        MP = ast.literal_eval(output.marginal_probs.data)
        return MP

    def get_G_bar(self, G, v, c_bar):
        v_bar = len(G.nodes())  # hypothesis node
        G_bar = copy.deepcopy(NN)
        G_bar.add_nodes_from([(v_bar, {'type': c_bar})])
        G_bar.add_edges_from([(v, v_bar)])

        return G_bar

    def FVE_v(self, G, v, P_E, K, M, N_c):
        # rollout 1
        MP_0 = self.srv_MBM(G, v)
        p_E_0 = sum([a*b for a, b in zip(MP_0[1:],P_E)])

        # rollout 2
        C_1 = np.argsort(MP_0)[::-1][0:M]
        G_1_list = []
        MP_1_list = []
        for c_bar in C_1:
            G_1 = self.get_G_bar(G, v, c_bar)
            v_1 = len(G_1.nodes()) - 1
            MP_1_c_bar = [a*MP_0[c_bar] for a in self.srv_MBM(G_1, v_1)]
            G_1_list.append(G_1)
            MP_1_list.append(MP_1_c_bar)
        MP_1 = [0 for i in range(N_c+1)]
        for MP_1_c_bar in MP_1_list:
            MP_1 = [a+b for a,b in zip(MP_1 , MP_1_c_bar)]
        p_E_1 = sum([a * b for a, b in zip(MP_1[1:], P_E)])

        # G_1_list.append(G_1)
        # for G_1 in G_1_list:
        #     v_1 = len(G_1.nodes()) - 1
        #     MP_1 = self.srv_MBM(G_1, v_1)
        #     print(G_1.nodes())
        #     print(nx.get_node_attributes(G_1, 'type'))
        #     print(MP_1)


NN = nx.Graph()
NN.add_nodes_from([(0, {'pos': (-28., 0.), 'type': 4, 'is_robot': True, 'to_go': False, 'value': 0}),
                   (1, {'pos': (-35., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (2, {'pos': (-28., 5.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.9}),
                   (3, {'pos': (-28., -5.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (4, {'pos': (-21., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1})])
NN.add_edges_from([(0, 1), (0,2), (0, 3), (0, 4)])


FVE = Frontier_Value_Estimation()
# p_E : [1xN_c] probability vector
p_E = [0.5, 0.1, 0.1, 0.1, 0.1]
# rollout length
K = 3

# high member cut
M = 3

# number of class
N_c = 5
FVE.FVE_v(NN, 0, p_E, K, M, N_c)








# # convert NN to dict
# dict_G = {}
# E = [[v1,v2] for (v1,v2) in list(nx.edges(NN))]
# C = {str(n):str(c) for n, c in nx.get_node_attributes(NN, 'type').items()}
# dict_G["edges"] = E
# dict_G["features"] = C
#
# msg_NN_jsonstr = String()
# msg_NN_jsonstr.data = str(dict_G)
# node = 1
#
# rospy.wait_for_service('pred_MBM')
# srv_pred_MBM = rospy.ServiceProxy('pred_MBM', pred_MBM)
# output = srv_pred_MBM(msg_NN_jsonstr, node)
# MP = ast.literal_eval(output.marginal_probs.data)
# print(MP)