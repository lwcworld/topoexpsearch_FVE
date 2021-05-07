#!/usr/bin/env python3

import networkx as nx
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from topoexpsearch_FVE.srv import pred_FVE
from std_msgs.msg import String
import ast
import numpy as np
import copy
from ast import literal_eval

class Frontier_Value_Estimation():
    def __init__(self):
        rospy.wait_for_service('pred_MBM', timeout=3)
        self.srv_pred_MBM = rospy.ServiceProxy('pred_MBM', pred_MBM)
        print('start')

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
        G_bar = copy.deepcopy(G)
        G_bar.add_nodes_from([(v_bar, {'type': c_bar})])
        G_bar.add_edges_from([(v, v_bar)])

        return G_bar

    def parse_msg(self, msg):
        NN_jsonstr = msg.NN_jsonstr.data # navigation network (json string type)
        NN_json = literal_eval(NN_jsonstr)
        G = nx.from_edgelist(NN_json["edges"])
        if "features" in NN_json.keys():
            features = NN_json["features"]
        features = {int(k): v for k, v in features.items()}

        for k, v in features.items():
            G.nodes[k]['type'] = v

        v = msg.node # interset node
        p_E = literal_eval(msg.p_E.data)
        N_c = msg.N_c
        M = msg.M
        K = msg.K
        gamma = msg.gamma
        return G, v, p_E, K, M, N_c, gamma

    def FVE_v(self, msg):
        G, v, p_E, K, M, N_c, gamma = self.parse_msg(msg)
        p_o = 0
        if K>=1:
            # rollout 1
            MP_0 = self.srv_MBM(G, v)
            p_E_0 = sum([a*b for a, b in zip(MP_0[1:],p_E)])
            p_o = p_o + p_E_0

        if K>=2:
            # rollout 2
            C_1 = np.argsort(MP_0)[::-1][0:M]
            G_1_list = []
            MP_1_list = []
            for c_bar in C_1:
                G_1_i = self.get_G_bar(G, v, c_bar)
                v_1_i = len(G_1_i.nodes()) - 1
                MP_1_c_bar = [a*MP_0[c_bar] for a in self.srv_MBM(G_1_i, v_1_i)]
                G_1_list.append(G_1_i)
                MP_1_list.append(MP_1_c_bar)
            MP_1 = [0 for i in range(N_c+1)]
            for MP_1_c_bar in MP_1_list:
                MP_1 = [a+b for a,b in zip(MP_1 , MP_1_c_bar)]
            p_E_1 = sum([a * b for a, b in zip(MP_1[1:], p_E)])

            p_o = p_o + gamma*p_E_1

        if K>=3:
            # rollout 3
            C_2_list = []
            G_2_list = [[] for i in range(M)]
            MP_2_list = [[] for i in range(M)]
            for i, (MP_1_i, G_1_i) in enumerate(zip(MP_1_list, G_1_list)):
                C_2_i = np.argsort(MP_1_i)[::-1][0:M]
                C_2_list.append(C_2_i)
                v = len(G_1_i.nodes())
                for c_bar in C_2_i:
                    G_2_i_j = self.get_G_bar(G_1_i, v, c_bar)
                    v_2_i_j = len(G_2_i_j.nodes()) - 1
                    MP_2_i_j_c_bar = [a * MP_1_i[c_bar] for a in self.srv_MBM(G_2_i_j, v_2_i_j)]
                    G_2_list[i].append(G_2_i_j)
                    MP_2_list[i].append(MP_2_i_j_c_bar)
            MP_2 = [0 for i in range(N_c+1)]
            for MP_2_i in MP_2_list:
                for MP_2_i_j in MP_2_i:
                    MP_2 = [a + b for a, b in zip(MP_2, MP_2_i_j)]
            p_E_2 = sum([a * b for a, b in zip(MP_2[1:], p_E)])

            p_o = p_o + gamma**2 * p_E_2

        return p_o

if __name__ == '__main__':
    rospy.init_node('abc')
    FVE = Frontier_Value_Estimation()

    s = rospy.Service('pred_FVE', pred_FVE, FVE.FVE_v)
    rospy.spin()