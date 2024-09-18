import tensorflow as tf

class data_generation:
    def __init__(self,
                 od_target=None,
                 ue_pathflows=None,
                 ue_linkflows=None,
                 link_target=None):

        # set input data sources
        self.od_target_data = od_target
        self.route_assignment_data = ue_pathflows
        self.link_data = ue_linkflows
        self.link_perform_data = link_target

        # origin (zonal) data
        self.ozone_df = self.route_assignment_data.groupby('o_zone_id')['volume'].sum()
        self.ozone_df = self.ozone_df.reset_index()
        self.ozone_df['o_node_id'] = self.ozone_df['o_zone_id']

        # origin-destination data
        self.od_df = self.route_assignment_data.groupby(['o_zone_id', 'd_zone_id'])[['volume', 'travel_time']].sum()
        self.od_df = self.od_df.reset_index()
        self.od_df['od_id'] = self.od_df.index + 1

        # check the length of target od flows is matched with the length of initial od flows and process the target od
        # flow data
        self.check_od_path_mapping_consistency()

        # get the ozone target data
        self.o_target_data = self.od_target_data.groupby('o_zone_id')['volume'].sum()
        self.o_target_data = self.o_target_data.reset_index()
        self.o_target_data['o_node_id'] = self.o_target_data['o_zone_id']

        # get the link target data
        self.link_target_data = self.link_perform_data[["link_id","vehicle_volume"]]
        self.link_target_data.rename(columns={"vehicle_volume": 'link_count_car_volumes'}, inplace=True)

        # path data
        self.path_df = self.route_assignment_data[['o_zone_id', 'd_zone_id', 'node_sequence', 'volume']]
        self.path_df['path_id'] = self.path_df.index + 1

        # link data
        self.link_data = self.link_data.merge(self.link_perform_data[["from_node_id", "to_node_id"]],
                                              on=["from_node_id", "to_node_id"], how="inner")
        self.link_df = self.link_data[[
            'link_id',
            'from_node_id',
            'to_node_id',
            "lanes",
            "lane_capacity",
            "fftt",
            'ref_volume',
            'distance_mile',
        ]]
        # self.link_df['car_vol'] = (self.link_df['ref_volume']).fillna(0)
        # FIXME: Check the truck link volumes
        # self.link_df['truck_vol'] = self.link_df['car_vol'] * 0.1
        self.link_df['link_no'] = self.link_df.index

        print(f"Number of Origins: {self.ozone_df.shape[0]}")
        print(f"Number of Origin-Destination Pairs: {self.od_df.shape[0]}")
        print(f"Number of Paths: {self.route_assignment_data.shape[0]}")
        print(f"Number of Links: {self.link_data.shape[0]}")

    def origin_layer(self):
        # origin layer
        node_zone_dict = self.ozone_df[['o_node_id', 'o_zone_id']].set_index('o_node_id').to_dict()['o_zone_id']
        return node_zone_dict

    def origin_dest_layer(self):
        # origin-destination layer
        node_zone_dict = self.origin_layer()
        self.od_df['od_pair'] = self.od_df.apply(lambda x: (int(x.o_zone_id), int(x.d_zone_id)), axis=1)
        self.od_df['o_zone_id'] = self.od_df.apply(lambda x: node_zone_dict[int(x.o_zone_id)], axis=1)
        od_pair_dict = self.od_df[['od_pair', 'od_id']].set_index('od_pair').to_dict()['od_id']
        return od_pair_dict

    def od_to_path_layer(self):
        od_pair_dict = self.origin_dest_layer()
        self.path_df['od_id'] = self.path_df.apply(lambda x: od_pair_dict[int(x.o_zone_id), int(x.d_zone_id)], axis=1)
        return self.path_df

    def path_link_layer(self):
        self.link_df['link_pair'] = self.link_df.apply(lambda x: (int(x.from_node_id), int(x.to_node_id)), axis=1)
        link_no_pair_dict = self.link_df[['link_no', 'link_pair']].set_index('link_pair').to_dict()['link_no']
        return link_no_pair_dict

    def incidence_mat(self):
        # To count the number of x_f flow variables
        link_no_pair_dict = self.path_link_layer()
        path_df = self.od_to_path_layer()
        path_no1 = 0
        od_path_idx_list = []
        path_link_idx_list = []
        init_path_vol = []
        for i in range(len(self.od_df)):
            od_id = self.od_df.loc[i, 'od_id']
            path_df_od = path_df[path_df['od_id'] == od_id].reset_index(drop=True)
            for j in range(len(path_df_od)):
                node_sequence = list(map(int, path_df_od.loc[j, 'node_sequence'].split(';')[0: -1]))
                link_sequence = [link_no_pair_dict[(node_sequence[k], node_sequence[k + 1])] for k in range(len(node_sequence) - 1)]

                init_path_vol.append(path_df['volume'][(path_df_od['path_id'] - 1)[j]])
                od_path_idx_list.append((i, path_no1))
                for link_id in link_sequence:
                    path_link_idx_list.append((path_no1, link_id))
                path_no1 += 1

                # if j < len(path_df_od) - 1:
                #     init_path_vol.append(path_df['volume'][(path_df_od['path_id'] - 1)[j]])
                #     od_path1_idx_list.append((i, path_no1))
                #     for link_id in link_sequence:
                #         path1_link_idx_list.append((path_no1, link_id))
                #     path_no1 += 1
                # else:
                #     od_path1_not_idx.append((i, path_no1))
                #     for link_id in link_sequence:
                #         path2_link_idx_list.append((i, link_id))

        return od_path_idx_list, path_link_idx_list, init_path_vol

    def o_od_incidence_mat(self):
        """
        Generate 1-dimensional list of mapping origin into origin-destination
        Returns:

        """
        o_od_idx_list = []
        # Index-Based Conversion
        # compare the index and o_ids...
        # FIXME: need to re-factorize the computation of incidence matrix - TEMPORARY CODE: non-sequence zonal ids
        o_od_num = 0
        self.ozone_df["index_o_zone"] = self.ozone_df.index
        self.od_df = self.od_df.merge(self.ozone_df[["o_zone_id", "index_o_zone"]], on="o_zone_id", how="outer")
        for o_id in self.ozone_df["index_o_zone"]:
            select_od = self.od_df[self.od_df["index_o_zone"] == o_id]
            for j in select_od["od_id"]:
                o_od_idx_list.append((o_id, o_od_num))
                o_od_num += 1

        return o_od_idx_list

    def reformed_incidence_mat(self):
        """
        Convert the incidence matrix into a sparse matrix for memory efficiency

        Returns:

        """
        od_path1_idx_list, path1_link_idx_list, init_path_flow = self.incidence_mat()
        num_link = self.link_df.shape[0]

        # TODO: use the initial origin-destination volume?
        # FIXME: od_volume: the initial OD flow volumes should be obtained by the path flow variables defined!!!
        #  Not Constant Values

        od_volume = tf.reshape(tf.constant(self.od_df['volume'], dtype=tf.float32), (-1, 1))
        spare_od_path_inc = tf.sparse.SparseTensor(
            od_path1_idx_list, [1.0] * len(od_path1_idx_list), (len(od_volume), od_path1_idx_list[-1][1] + 1))

        path_link_inc = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(
            path1_link_idx_list, [1.0] * len(path1_link_idx_list), (path1_link_idx_list[-1][0] + 1, num_link))))
        # path_link_inc_n = tf.sparse.to_dense(tf.sparse.reorder(tf.sparse.SparseTensor(
        #     path2_link_idx_list, [1.0] * len(path2_link_idx_list), (path2_link_idx_list[-1][0] + 1, num_link))))

        # return od_volume, spare_od_path_inc, path_link_inc, path_link_inc_n, init_path_flow
        return od_volume, spare_od_path_inc, path_link_inc, init_path_flow

    def get_o_to_od_incidence_mat(self):
        o_od_idx_list = self.o_od_incidence_mat()
        sparse_o_od_inc = tf.sparse.SparseTensor(
            o_od_idx_list, [1.0] * len(o_od_idx_list), (len(self.ozone_df), o_od_idx_list[-1][1] + 1))

        return sparse_o_od_inc

    def get_init_path_values(self):
        _, _, _, init_path_flow = self.reformed_incidence_mat()

        path_flow = tf.Variable(init_path_flow, dtype=tf.float32)  # Set UE Path Flows initially Estimated by DTALite

        return path_flow

    def get_bpr_params(self):
        bpr_params = {}
        # FIXME: NAME CONSISTENCY ("VDF_fftt" or fftt?)
        bpr_params["fftt"] = tf.reshape(tf.constant(self.link_df['fftt'], dtype=tf.float32), (-1, 1))

        # FIXME: NAME CONSISTENCY ("capacity" or lane_capacity"?)
        bpr_params["cap"] = tf.reshape(tf.constant(self.link_df["lane_capacity"] \
                                                   * self.link_df["lanes"], dtype=tf.float32), (-1, 1))
        bpr_params["alpha"] = 0.15
        bpr_params["beta"] = 4

        return bpr_params

    def get_lagrangian_params(self, path_link_inc):

        lagrangian_params = {}
        lagrangian_params["rho_factor"] = 1.0
        lagrangian_params["lambda_factor"] = 1.0
        init_lambda_values = tf.zeros((len(path_link_inc), 1), tf.float32)

        return lagrangian_params, init_lambda_values

    def check_od_path_mapping_consistency(self):
        init_od_df = self.od_df
        target_od_df = self.od_target_data

        if init_od_df.shape[0] != target_od_df.shape[0]:
            print(f"WARNING: The length of target OD flows is {target_od_df.shape[0]}, "
                  f"but the length of initial OD flows is {init_od_df.shape[0]}. ")
            print()
            print("Imputing the od flow target data to remove unmapped OD pairs...")
            print(f" - Before removing, the unmapped target od volume is {self.od_target_data['volume'].sum()}", )
            self.od_target_data = \
                target_od_df.merge(init_od_df[["o_zone_id", "d_zone_id", "od_id"]], on=["o_zone_id", "d_zone_id"], how="inner")
            print(f" - After processing, the target od volume is {self.od_target_data['volume'].sum()}")
            return self.od_target_data
        else:
            print("The lengths of target and initial OD flows match, no action required.")

            return None