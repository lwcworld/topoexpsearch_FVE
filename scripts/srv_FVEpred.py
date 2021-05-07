#!/usr/bin/env python3
import os
print(os.getcwd())
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from gensim.models.doc2vec import Doc2Vec
from keras.models import model_from_json
from Utils_MBM import *
from Utils_graph2datapair import *
from Utils_graph2vec import *
from std_msgs.msg import String
from ast import literal_eval


class cb_srv():
    def __init__(self):
        pass

    def handle_predMBM(self, msg):
        NN_jsonstr = msg.nav_net.data # navigation network (json string type)
        s = msg.node # interset node

        NN_json = literal_eval(NN_jsonstr)
        NN = nx.from_edgelist(NN_json["edges"])
        if "features" in NN_json.keys():
            features = NN_json["features"]
        else:
            features = nx.degree(NN)
        features = {int(k): v for k, v in features.items()}

        for k, v in features.items():
            NN.nodes[k]['type'] = v


        # param
        dim = 8
        N_class = 5
        level = 4

        # path
        dir_package = '/home/lwcubuntu/workspaces/topoexpsearch/src/topoexpsearch_MBM/'
        path_embedding_model = dir_package + 'dataset/model/graph_embedding/model_embedding_dim' + str(dim)
        path_MBM_architecture = dir_package + 'dataset/model/MBM/arc_lv4_iter0.json'
        dir_param = dir_package + 'dataset/param_c5/'

        # load graph2vec model
        model_embedding = Doc2Vec.load(path_embedding_model)

        # load MBM model
        with open(path_MBM_architecture, 'r') as json_file:
            model_MBM = model_from_json(json_file.read(), custom_objects={'exp_advanced': exp_advanced})

        # load MBM model params
        Tau_binary = np.load(dir_param + 'Tau_binary_lv' + str(level) + '.npy')
        subset_tau = np.load(dir_param + 'subset_tau_lv' + str(level) + '.npy')

        # preprocess networkx NN
        # add hypothetical node
        h = len(NN.nodes())  # hypothesis node
        NN_H = copy.deepcopy(NN)
        NN_H.add_nodes_from([(h, {'type': 0})])
        NN_H.add_edges_from([(s, 5)])

        # get node feature (type)
        feature = nx.get_node_attributes(NN_H, 'type')
        for k, v in feature.items():
            feature[k] = str(v)

        # get tree graph & feature
        tree_graph, feature = get_bfstree(NN_H, feature, root=h, depth_limit=3)

        # decompose
        dG_list, df_list = decompose_tree_graph(tree_graph, feature)

        # graph2vec predict
        wl_iterations = 2
        vec_array = np.zeros((len(dG_list), dim))
        for iter, (dG, df) in enumerate(zip(dG_list, df_list)):
            machine = WeisfeilerLehmanMachine(dG, df, wl_iterations)
            wl_feature = TaggedDocument(words=machine.extracted_features, tags=["g_" + '0'])
            vec = model_embedding.infer_vector(wl_feature[0])
            vec_array[iter] = vec

        mp_d = get_predicted_prob(model_MBM, vec_array, Tau_binary, subset_tau, N_class)
        mp_o = np.mean(mp_d, 0)

        output = String()
        output.data = str(mp_o)

        return output

if __name__ == '__main__':
    rospy.init_node('MBM_server')
    cb_srv = cb_srv()

    s = rospy.Service('pred_MBM', pred_MBM, cb_srv.handle_predMBM)
    rospy.spin()