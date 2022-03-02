CUDA_VISIBLE_DEVICES=0 python3 ./feature_precomputation_readmission.py   --task_name readmission   --readmission_mode discharge  --data_dir /PatientTM/data/extended_folds/discharge_subjectsplit_trajectory/   --bert_model /PatientTM/results/result_discharge/2022-01-17_15:03:46_polar-shape-885/ --max_seq_length 512   --output_dir /PatientTM/results/result_discharge/ -feat daystoprevadmit duration


CUDA_VISIBLE_DEVICES=0 python3 ./feature_precomputation_diagnosis.py --task_name diagnosis_prediction  --readmission_mode discharge  --data_dir /PatientTM/data/extended_folds/discharge_subjectsplit_trajectory/ --bert_model /PatientTM/results/result_discharge/2022-01-11_10:37:47_unique-frost-840/   --max_seq_length 512   --output_dir /PatientTM/results/result_discharge --codes_to_predict small_diag_icd9  -feat small_diag_icd9 daystoprevadmit


CUDA_VISIBLE_DEVICES=0 python3 ./feature_precomputation_diagnosis.py --task_name diagnosis_prediction  --readmission_mode discharge  --data_dir /PatientTM/data/extended_folds/discharge_subjectsplit_trajectory/ --bert_model /PatientTM/results/result_discharge/2022-01-16_20:45:49_breezy-elevator-876/   --max_seq_length 512   --output_dir /PatientTM/results/result_discharge --codes_to_predict diag_ccs  -feat diag_ccs daystoprevadmit

