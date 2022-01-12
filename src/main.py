import argparse

from readmission.run_model import runReadmission
from diagnosis_prediction.run_model import runDiagnosisPrediction


def help(show=False):
    parser = argparse.ArgumentParser(description="")

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    
    parser.add_argument("--readmission_mode", default = None, type=str, help="early notes or discharge summary")
    
    parser.add_argument("--task_name",
                        default=None,
                        required=True,
                        type=str,
                        choices=["readmission", "diagnosis_prediction"],
                        help="The name of the task to run. Please select one of the following predictive tasks: [readmission, diagnosis_prediction].")    
    parser.add_argument("--codes_to_predict",
                        default=None,
                        required=False,
                        type=str,
                        choices=["small_diag_icd9", "diag_ccs"],
                        help="The type of code to predict in code predicting tasks. Please select one of the following predictive tasks: [small_diag_icd9, diag_ccs].")    
    
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run model training.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run a testing phase on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--test_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('-feat','--features',
                        default=None,
                        nargs="*",
                        type=str,
                        choices=["clinical_text", "admittime", "daystonextadmit", "daystoprevadmit", "duration", "diag_ccs", "proc_ccs", "small_diag_icd9",
                                "small_proc_icd9", "cui"], #"proc_icd9", "diag_icd9", "ndc"],
                        help='Features to use as model input. Please select one or more of the following inputs: [clinical_text, admittime, daystonextadmit, daystoprevadmit, duration, diag_ccs, proc_ccs, small_diag_icd9, small_proc_icd9, cui]')
    parser.add_argument('--small_icd9_ccs_maxlength',
                        type=int,
                        default=39,
                        help="max length for icd9 and ccs tensors")
    parser.add_argument('--cui_maxlength',
                        type=int,
                        default=47,
                        help="max length for cui tensors")
    parser.add_argument('--freeze_bert',
                        default=False,
                        action='store_true',
                        help="Whether to freeze parameters from BERT layers or not. When frozen, these are not updated during model training.")
    parser.add_argument('--early_stop',
                        default=False,
                        action='store_true',
                        help="Save a model checkpoint using early stopping to prevent the saving of overfiting models.")
    parser.add_argument('--subsampling',
                        default=False,
                        action='store_true',
                        help="Subsample the training datasets to equalize the distribution of positive vs negative classes. Useful for readmission prediction.")

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
    # elif args.task_name == "code_prediction":
    #     runCodePrediction(args)

    
main()

