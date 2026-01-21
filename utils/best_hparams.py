mortality_day90_best_hparams = {
    "dis_coeff": 0.02,
    "dis_lr": 0.000890933949606578,
    "dis_step": 2,
    "lr": 0.000691457262659161,
    'batch_size': 64}


readmission_day30_best_hparams = {
    "dis_coeff": 0.005,
    "dis_lr": 3.390697073884055e-05,
    "dis_step": 1,
    "lr": 0.0007018482904541707,
    'batch_size': 64}


icu_mortality_best_hparams = {
    "dis_coeff": 0.02,
    "dis_lr": 0.0007314441336154807,
    "dis_step": 4,
    "lr": 0.000673464810428309,
    'batch_size': 64}


remaining_los_best_hparams = {
    "dis_coeff": 0.1,
    "dis_lr": 0.0005082563101808861,
    "dis_step": 2,
    "lr": 8.625660619309187e-05,
    'batch_size': 64}


mimic_graph_embedding_best_hparams = {
    'batch_size': 128,
    'd': 32,
    'lr': 0.0009645408090711602,
    'r': 128
}

eicu_graph_embedding_best_hparams = {
    'batch_size': 32,
    'd': 64,
    'lr': 0.00043727576823359674,
    'r': 64
}

mimic_pretrain_best_hparams = {
    "lr": 0.0009447560809539598,
    "batch_size": 32,
}

eicu_pretrain_best_hparams = {
    "lr": 0.0005415928190945238,
    "batch_size": 32,
}