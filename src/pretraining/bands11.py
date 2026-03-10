# Band mapping for PRETRAINING (no CloudSat overpass mask)
# 0-based indices → common band order
BAND_MAPPING = {
    "GOES": [1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],      # 20 bands, no cloudsat
    "HIMAWARI": [2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],  # 20 bands, no cloudsat
    "MSG": [9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14],           # 15 bands, no cloudsat
}

BAND_NAMES = [
    "red", "nir", "swir_1.6", "swir_3.9",
    "wv_6.2", "wv_7.3", "cloud_phase", "ozone",
    "ir_10.5", "ir_11.2", "co2",
    "sat_zen", "sat_azi", "sol_zen", "sol_azi"
]  # 15 common bands (no cloudsat_mask for pretraining)

BAND_TYPES = ["nr" for _ in range(3)]+["bt" for _ in range(8)]+ ["angle_zen","angle_azi","angle_zen","angle_azi"]

