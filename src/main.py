import argparse

from readmission.run_model import runReadmission
from diagnosis_prediction.run_model import runDiagnosisPrediction
from trajectory_modelling.run_model import runTrajectoryModelling


def help(show=False):
    parser = argparse.ArgumentParser(description="")

    runningMode = parser.add_argument_group('Data Settings', 'Parameters to define model input features, data directories (input and output), etc. Only --features, --data_dir and --output_dir are mandatory.')
    runningMode.add_argument("--task_name",
                             default=None,
                             required=True,
                             type=str,
                             choices=["readmission", "diagnosis_prediction", "trajectory_modelling"],
                             help="The name of the task to run. Please select one of the following predictive tasks: [readmission, diagnosis_prediction,\
                             trajectory_modelling].")
    runningMode.add_argument("--trajectory_subtask_name",
                             default=None,
                             #required=True,
                             type=str,
                             choices=["readmission", "diagnosis"],
                             help="The name of the subtask to run in the trajectory model. Only necessary when the selected --task_name is trajectory_modelling.\
                             Please select one of the following predictive tasks: [readmission, diagnosis].")   
    runningMode.add_argument("--codes_to_predict",
                             default=None,
                             type=str,
                             choices=["small_diag_icd9", "diag_ccs"],
                             help="Type of code to predict in code predicting tasks. Please select one of the following predictive tasks: [small_diag_icd9, diag_ccs].")   
    runningMode.add_argument("--do_train",
                             default=False,
                             action='store_true',
                             help="Run the model in development mode.")
    runningMode.add_argument("--do_test",
                             default=False,
                             action='store_true',
                             help="Run the model in inference mode.")
    
    
    inputFeatures = parser.add_argument_group('Data Settings', 'Parameters used to define model input features, data directories (input and output), etc. The --features, --data_dir and --output_dir are mandatory.')
    inputFeatures.add_argument('-feat','--features',
                               default=None,
                               nargs="*",
                               type=str,
                               required=True,
                               choices=["clinical_text", "daystoprevadmit", "duration", "diag_ccs", "proc_ccs", "small_diag_icd9",
                                        "small_proc_icd9", "cui"],
                               help='Features to use as model input. Clinical text representations can be either obtained in run-time or from precomputed files.\
                               Please select one or more of the following inputs: [clinical_text, daystoprevadmit, duration, diag_ccs, proc_ccs, small_diag_icd9, small_proc_icd9, cui]')
    inputFeatures.add_argument("--data_dir",
                               default=None,
                               type=str,
                               required=True,
                               help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    inputFeatures.add_argument("--output_dir",
                               default=None,
                               type=str,
                               required=True,
                               help="The output directory where model checkpoints and other results shall be written to.")
    inputFeatures.add_argument("--precomputed_text",
                               default=False,
                               action='store_true',
                               help="Whether to use precomputed representations for clinical text which have been priorly saved to disk. Only important when using clinical_text as input.")
    inputFeatures.add_argument("--readmission_mode",
                               default = None,
                               type=str,
                               choices=["discharge", "early_2days", "early_3days"],
                               help="The type of notes being used: early notes or discharge summary notes.")
    inputFeatures.add_argument('--small_icd9_ccs_maxlength',
                               type=int,
                               default=39,
                               help="max length for icd9 and ccs tensors")
    inputFeatures.add_argument('--cui_maxlength',
                               type=int,
                               default=47,
                               help="max length for cui tensors")
    
    
    bertSettings = parser.add_argument_group('ClinicalBERT Settings', 'System parameters for the clinicalBERT model. Important when using clinical text as input feature.')
    bertSettings.add_argument("--bert_model", default=None, type=str,
                              #required=True,
                              help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual,\
                              bert-base-chinese.")
    bertSettings.add_argument("--max_seq_length",
                              default=128,
                              type=int,
                              help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
    bertSettings.add_argument('--bert_finetune_epochs',
                              type=int,
                              default=0,
                              help="Number of epochs to finetune the clinicalBERT model. Default is 0, acting as a frozen model.")
    bertSettings.add_argument('--bert_num_trainable_layers',
                              type=int,
                              default=0,
                              help="Number of clinicalBERT layers to be finetuned (from end to beginning of the model). Default is 0, acting as a frozen model.\
                              Eg: 1 corresponds to finetuning the last layer.")
    

    developmentSettings = parser.add_argument_group('Model Development Settings', 'System parameters for model development and inference.')
    developmentSettings.add_argument("--train_batch_size",
                                     default=32,
                                     type=int,
                                     help="Total batch size for training.")
    developmentSettings.add_argument("--test_batch_size",
                                     default=2,
                                     type=int,
                                     help="Total batch size for test.")
    developmentSettings.add_argument("--learning_rate",
                                     default=5e-5,
                                     type=float,
                                     help="The initial learning rate for Adam.")
    developmentSettings.add_argument("--weight_decay",
                                     default=1e-4,
                                     type=float,
                                     help="The initial weight decay for the optimizer.")
    developmentSettings.add_argument("--num_train_epochs",
                                     default=3.0,
                                     type=float,
                                     help="Total number of training epochs to perform.")
    developmentSettings.add_argument("--warmup_proportion",
                                     default=0.1,
                                     type=float,
                                     help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    developmentSettings.add_argument("--no_cuda",
                                     default=False,
                                     action='store_true',
                                     help="Whether not to use CUDA when available")
    developmentSettings.add_argument("--local_rank",
                                     type=int,
                                     default=-1,
                                     help="Used for distributed training on gpus")
    developmentSettings.add_argument('--seed', 
                                     type=int, 
                                     default=42,
                                     help="Random seed for initialization")
    developmentSettings.add_argument('--gradient_accumulation_steps',
                                     type=int,
                                     default=1,
                                     help="Number of update steps to accumulate before performing a backward/update pass.")                       
    developmentSettings.add_argument('--optimize_on_cpu',
                                     default=False,
                                     action='store_true',
                                     help="Whether to perform optimization and keep the optimizer averages on CPU")
    developmentSettings.add_argument('--fp16',
                                     default=False,
                                     action='store_true',
                                     help="Whether to use 16-bit float precision instead of 32-bit")
    developmentSettings.add_argument('--loss_scale',
                                     type=float, default=128,
                                     help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    developmentSettings.add_argument('--early_stop',
                                     default=False,
                                     action='store_true',
                                     help="Save a model checkpoint using early stopping to prevent the saving of overfiting models.")
    developmentSettings.add_argument("--patience",
                                     default=2000,
                                     type=int,
                                     help="Patience (number of steps) for early stop before breaking training loop.")
    developmentSettings.add_argument('--subsampling',
                                     default=False,
                                     action='store_true',
                                     help="Subsample the training datasets to equalize the distribution of positive vs negative classes.\
                                     Currently only supported in readmission prediction.")
    
    
    trajectorySettings = parser.add_argument_group('Trajectory-based Settings', 'Model parameters for the trajectory-based model.')
    trajectorySettings.add_argument('--recurrent_hidden_size',
                                    type=int,
                                    default=100,
                                    help="Hidden size for the recurrent network.")
    trajectorySettings.add_argument('--recurrent_num_layers',
                                    type=int,
                                    default=1,
                                    help="Number of layers in the recurrent network.")
    trajectorySettings.add_argument("--bidirectional",
                                    default=False,
                                    action='store_true',
                                    help="Whether to have a bidirectional recurrent model.")
    trajectorySettings.add_argument("--recurrent_network",
                                    default=None,
                                    type=str,
                                    choices=["LSTM", "GRU"],
                                    help="The type of recurrent network to use. Please select one of the following: [GRU, LSTM].")
    trajectorySettings.add_argument("--visit_sliding_window",
                                    default=None,
                                    type=str,
                                    choices=["3", "6"],
                                    help="The number of visits to consider in the sliding window. Please select one of the following: [3, 6].")
    trajectorySettings.add_argument("--multi_hot_diag",
                                    default=False,
                                    action='store_true',
                                    help="Whether to use multi hot diagnoses input instead of embeddings or not.")
    
    if show:
        parser.print_help()
    return parser.parse_args()
    
    
def main():
    args = help()
    
    if not args.task_name:
        print("Nothing to do, please define the execution mode!")
        help(show=True)
        exit()
        
    if args.task_name == "readmission":
        runReadmission(args)
    elif args.task_name == "diagnosis_prediction":
        runDiagnosisPrediction(args)
    elif args.task_name == "trajectory_modelling":
        runTrajectoryModelling(args)

    
main()

