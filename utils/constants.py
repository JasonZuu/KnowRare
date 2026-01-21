
MIMIC_RARE_ICD_CODES =["ICD_117", "ICD_280", "ICD_201", "ICD_235", "ICD_494", 
                       "ICD_054", "ICD_141", "ICD_239", "ICD_991", "ICD_874"]

EICU_RARE_ICD_CODES =['ICD_801', 'ICD_729', 'ICD_156', 'ICD_171', 'ICD_284',
                      'ICD_991', 'ICD_420', 'ICD_205', 'ICD_322', 'ICD_513']

COMMON_ICD_CODES = ['ICD_038']

MIMIC_TS_FEATURES = [
    "heartrate_min", "heartrate_max", "heartrate_mean",
    "sysbp_min", "sysbp_max", "sysbp_mean",
    "diasbp_min", "diasbp_max", "diasbp_mean",
    "meanbp_min", "meanbp_max", "meanbp_mean",
    "resprate_min", "resprate_max", "resprate_mean",
    "tempc_min", "tempc_max", "tempc_mean",
    "spo2_min", "spo2_max", "spo2_mean",
    "glucose_min", "glucose_max", "glucose_mean",
    "ALBUMIN", "ANION GAP", "BANDS", "BICARBONATE", 
    "BILIRUBIN", "BUN", "CHLORIDE", "CREATININE", 
    "GLUCOSE", "HEMATOCRIT", "HEMOGLOBIN", "INR", 
    "LACTATE", "PLATELET", "POTASSIUM", "PT", 
    "PTT", "SODIUM", "WBC"
]

task_label_dict = {
    "readmission_day30": "days_30_readmission_flag",
    "mortality_day90": "days_90_expire_flag",
    'icu_mortality': "icu_mortality",
    "remaining_los": "remaining_los",
}