import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from .data import data_generation
from .calibration import odme_mapping_variables, optimization
import logging

warnings.filterwarnings('ignore')  # ignore warning messages
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)  # configuration of logging messages


def run_optimization(od_volume: tf.Tensor,
                     sparse_matrix: dict,
                     path_link_inc: tf.Tensor,
                     path_flow: tf.Tensor,
                     bpr_params: dict,
                     optimization_params: dict,
                     target_data: dict,
                     obj_setting: dict,
                     access_data_sources,
                     output_path,
                     ) -> None:
    """
    Compute multi-objective optimization loss functions to adjust user equilibrium path flows

    Parameters:
    - od_volume (tf.Tensor): Origin-Destination (OD) volume matrix.
    - sparse_matrix (dict): Sparse incidence matrix (e.g., mapping origin-destination into paths).
    - path_link_inc (tf.Tensor): Path-link incidence matrix.
    - path_flow (tf.Tensor): Initial path flows.
    - bpr_params (dict): Parameters for the BPR (Bureau of Public Roads) function.
    - training_steps (int): Number of ADMM training steps.
    - target_data (dict): observed target data
    - init_path_flows (tf.Tensor): initialized path flows (either DTALite results or randomly generated)
    - obj_setting (dict): dictionary of multi-objective dispersion parameters
    - access_data_sources (classmethod): class that accesses loaded data sources
    Returns:
    None
    """

    logging.info("Finding Optimal Path Flows...")
    init_odme_mapping_variables = {}
    # get the initialized odme mapping variables
    init_link_volumes, init_link_costs, init_od_flows, init_o_flows, _ = odme_mapping_variables(path_flow,
                                                                                               sparse_matrix,
                                                                                               path_link_inc,
                                                                                               bpr_params)
    init_odme_mapping_variables["link_counts"] = init_link_volumes
    init_odme_mapping_variables["link_costs"] = init_link_costs
    init_odme_mapping_variables["od_flows"] = init_od_flows
    init_odme_mapping_variables["o_flows"] = init_o_flows

    # start to find optimal path flows
    optimal_path_flow, losses = optimization(path_flow,
                                             bpr_params,
                                             od_volume,
                                             sparse_matrix,
                                             path_link_inc,
                                             target_data,
                                             init_odme_mapping_variables,
                                             optimization_params,
                                             obj_setting,
                                             access_data_sources,
                                             )
    # get the odme mapping variables using the optimal path flows
    estimated_link_volumes, estimated_link_costs, estimated_od_flows, estimated_o_flows, optimal_paths = \
        odme_mapping_variables(
                                optimal_path_flow,
                                sparse_matrix,
                                path_link_inc,
                                bpr_params
        )


    concat_path_flows = get_path_flow_columns(access_data_sources, optimal_paths)
    link_dist = np.array(access_data_sources.link_df["distance_mile"], dtype="f")
    evaluation(losses,
               concat_path_flows,
               tf.squeeze(estimated_link_volumes),
               tf.squeeze(estimated_link_costs),
               tf.squeeze(estimated_od_flows),
               tf.squeeze(estimated_o_flows),
               target_data,
               link_dist,
               access_data_sources,
               output_path,
               )
    logging.info("Complete!")


def get_path_flow_columns(data_source, optimal_paths):
    """

    Args:
        data_source:
        optimal_paths:

    Returns:

    """
    od_layer = data_source.od_df
    path_layer = data_source.od_to_path_layer()
    restore_path_flows = []
    multi_od_pair_idx = 0

    for i in range(len(od_layer)):
        od_id = od_layer.loc[i, 'od_id']
        od_path = path_layer[path_layer['od_id'] == od_id].reset_index(drop=True)

        for j in range(len(od_path)):
            restore_path_flows.append(optimal_paths["multi_od_pairs"][multi_od_pair_idx].numpy().item())
            multi_od_pair_idx += 1

    return restore_path_flows

def rmse(estimated, observation):
    """

    Args:
        estimated:
        observation:

    Returns:

    """
    squared_diff = (estimated - observation) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rmse_value = np.sqrt(mean_squared_diff)
    return rmse_value


def evaluation(losses,
               optimal_path_flows,
               estimated_link_volumes,
               estimated_link_costs,
               estimated_od_flows,
               estimated_o_flows,
               target_data,
               link_dist,
               access_data_sources,
               output_path,
               ):
    """

    Args:
        losses:
        optimal_path_flows:
        estimated_link_volumes:
        estimated_od_flows:
        estimated_o_flows:
        target_data:
        data_imputation:

    Returns:

    """
    logging.info("Saving loss ...")
    get_df_losses = pd.DataFrame(losses, columns=["losses"])
    get_df_losses.index.name = "epoch"

    output_dir = output_path

    # Check if the folder exists
    if not os.path.exists(output_dir):
        # Create the folder
        os.makedirs(output_dir)

    get_df_losses.to_csv(output_dir + "loss_results.csv")

    logging.info("Saving the calibrated o flows ...")
    load_o_df = access_data_sources.o_target_data
    load_o_df["est_o_flows"] = estimated_o_flows
    load_o_df.to_csv(output_dir + "calibrated_o_flows.csv", index=False)

    logging.info("Saving the calibrated od flows ...")
    load_od_df = access_data_sources.od_target_data
    load_od_df["est_od_flows"] = estimated_od_flows
    load_od_df.to_csv(output_dir + "calibrated_od_flows.csv", index=False)

    logging.info("Saving the calibrated path flows ...")
    # FIXME: data column name consistency (#path_no or route_seq_id? and "link_id_sequence" or "link_sequence)
    load_path_df = access_data_sources.route_assignment_data[["path_no",
                                                    "o_zone_id",
                                                    "d_zone_id",
                                                    "node_sequence",
                                                    "link_sequence",
                                                    "geometry",
                                                    ]]
    path_flow_df = pd.DataFrame(optimal_path_flows, columns=["Path_Flows"])
    load_path_df = load_path_df.join(path_flow_df)
    load_path_df.to_csv(output_dir + "calibrated_path_flows.csv", index=False)

    logging.info("Saving the calibrated link flows ...")
    # FIXME: data column name consistency ("VDF_fftt or fftt" and "capacity" or lane_capacity"?)
    load_link_df = access_data_sources.link_data[[
        "link_id",
        "from_node_id",
        "to_node_id",
        "ref_volume",
        "fftt",
        "lane_capacity",
        "lanes",
    ]]
    link_flow_df = pd.DataFrame(estimated_link_volumes, columns=["est_link_flows"])
    load_link_df = load_link_df.join(link_flow_df)
    load_link_df.to_csv(output_dir + "calibrated_link_flows.csv", index=False)

    # Performance validation using RMSE
    rmse_car_link_volumes = rmse(estimated_link_volumes, target_data["link_count_car"])
    # rmse_truck_link_volumes = rmse(estimated_link_volumes, target_data["link_count_truck"])
    rmse_car_vmt = rmse(np.sum(estimated_link_volumes * link_dist),
                        target_data["VMT_car"])
    # rmse_truck_vmt = rmse(np.sum(estimated_link_volumes * link_dist),
    #                         target_data["VMT_truck"])
    # rmse_car_vht = rmse(np.sum(estimated_link_costs),
    #                     target_data["VHT_car"])
    # rmse_truck_vht = rmse(np.sum(estimated_link_costs),
    #                       target_data["VHT_truck"])
    rmse_od_flows = rmse(estimated_od_flows, target_data["observed_od_volume"])
    rmse_o_flows = rmse(estimated_o_flows, target_data["observed_o_volume"])

    # logging messages
    logging.info(f"RMSE: Passenger Car Count: {rmse_car_link_volumes}")
    logging.info(f"RMSE: Passenger Car VMT: {rmse_car_vmt}")
    # logging.info(f"RMSE: Passenger Car VHT: {rmse_car_vht}")
    # logging.info(f"RMSE: Truck Count: {rmse_truck_link_volumes}")ß
    # logging.info(f"RMSE: Truck VMT: {rmse_truck_vmt}")
    # logging.info(f"RMSE: Truck VHT: {rmse_truck_vht}")
    logging.info(f"RMSE: OD Flow: {rmse_od_flows}")
    logging.info(f"RMSE: O Flow: {rmse_o_flows}")