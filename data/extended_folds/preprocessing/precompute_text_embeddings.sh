CUDA_VISIBLE_DEVICES=0 python3 ./cbert_precomputation.py   --task_name readmission   --readmission_mode discharge  --data_dir /PatientFM/data/extended_folds/discharge/   --bert_model /PatientFM/model/pretraining   --max_seq_length 512   --output_dir /PatientFM/results/result_discharge

CUDA_VISIBLE_DEVICES=0 python3 ./cbert_precomputation.py   --task_name readmission   --readmission_mode early  --data_dir /PatientFM/data/extended_folds/early/   --bert_model /PatientFM/model/pretraining   --max_seq_length 512   --output_dir /PatientFM/results/result_early



CUDA_VISIBLE_DEVICES=0 python3 ./cbert_precomputation.py   --task_name readmission   --readmission_mode discharge_subjectsplit  --data_dir /PatientFM/data/extended_folds/discharge_subjectsplit/   --bert_model /PatientFM/model/pretraining   --max_seq_length 512   --output_dir /PatientFM/results/result_discharge

CUDA_VISIBLE_DEVICES=0 python3 ./cbert_precomputation.py   --task_name readmission   --readmission_mode early_subjectsplit  --data_dir /PatientFM/data/extended_folds/early_subjectsplit/   --bert_model /PatientFM/model/pretraining   --max_seq_length 512   --output_dir /PatientFM/results/result_early
