# Band mapping for PRETRAINING (no CloudSat overpass mask)
# 0-based indices → common band order
BAND_MAPPING = {
    "GOES": [1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],      # 20 bands,  
    "Himawari": [2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,20],  # 20 bands, 
    "MSG": [9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14,15],           # 15 bands, 
}

BAND_NAMES = [
    "red", "nir", "swir_1.6", "swir_3.9",
    "wv_6.2", "wv_7.3", "cloud_phase", "ozone",
    "ir_10.5", "ir_11.2", "co2",
    "sat_zen", "sat_azi", "sol_zen", "sol_azi"
]  # 15 common bands (no cloudsat_mask for pretraining)

BAND_TYPES = ["nr" for _ in range(3)]+["bt" for _ in range(8)]+ ["angle_zen","angle_azi","angle_zen","angle_azi"]

CS_VARS = ['CPR_Cloud_Mask', # - Radar Cloud Detection',
            'Radar_Reflectivity', # - W-band Backscatter (dBZ)',
            'Cloud_Type_Mask', # - Hydrometeor Classification',
            'Effective_Radius', # - Particle Size (μm)',
            'Ice_Water_Content', # - Frozen Hydrometeor Mass (g/m³)',
            'Liquid_Water_Content', # - Unfrozen Hydrometeor Mass (g/m³)',
            'Ice_Water_Content_RO', # - Retrieved Ice Mass (g/m³)',
            'Atmospheric_Height'] # - Vertical Level (m)']]

CS_BAND_MAPPING = [1, 3, 4]  # "radar reflectivity", "effective_radius", "ice_water_content"

CS_VARS = [CS_VARS[i] for i in CS_BAND_MAPPING]

CLOUDSAT_NAN_FILL_VALUES = {
    "Radar_Reflectivity": -35,
    "CloudTypeMask": 0,
    "CPR_Cloud_mask": 0,
    "RO_liq_effective_radius": 0,
    "RO_ice_effective_radius": 0,
    "RO_liq_number_conc": 0,
    "RO_ice_number_conc": 0,
    "RO_liq_water_content": 0,
    "RO_ice_water_content": 0,
    "Effective_Radius": 0,
    "Ice_Water_Content": 0,
    "QR": 0,
}