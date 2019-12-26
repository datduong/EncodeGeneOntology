
from argparse import ArgumentParser
from pathlib import Path

# start_near_true_label = False ## global


def get_args():
  parser = ArgumentParser(
      description='encode label descriptions into vectors')

  parser.add_argument('--lr', type=float, default=0.0001,
                      help='learning rate')
  parser.add_argument('--result_folder', type=str, default=None,
                      help='where to save result')
  parser.add_argument('--epoch', type=int, default=200,
                      help='num of iteration')
  parser.add_argument('--def_emb_dim', type=int, default=300,
                      help='emb dim for definition of GO terms')
  parser.add_argument('--model_load', type=str, default=None,
                      help='path to a model to load')
  parser.add_argument('--w2v_emb', type=str, default=None,
                      help='path to a w2v emb')
  parser.add_argument('--not_train_w2v_emb', action='store_true',
                      help='do not train word emb')
  parser.add_argument('--do_test', action='store_true',
                      help='do_test only')
  parser.add_argument('--metric_option', type=str, required=True,
                      help='cosine or entailment')

  parser.add_argument('--layer_index', type=int, default=None,
                      help='-1 is last -2 is 2nd last')

  parser.add_argument('--average_layer', action='store_true', default=False,
                      help='allows user to use second to last layer in the BERT model as vector')

  parser.add_argument('--write_score', type=str, default=None,
                      help='write out score for 2 GO vectors')

  parser.add_argument('--test_file', type=str, default=None,
                      help='some test file not have to be test set')

  parser.add_argument('--pretrained_label_emb', type=str, default=None,
                      help='pretrained_label_emb for gcnn')
  parser.add_argument('--do_continue', action='store_true',
                      help='continue training only')
  parser.add_argument('--reduce_cls_vec', action='store_true',
                      help='768 to smaller dim then compare the cosine')
  parser.add_argument('--update_bert', action='store_true',
                      help='update bert model, do this rarely')

  parser.add_argument("--bert_model",
                      default=None,
                      type=str,
                      help="Where pretrained bert are saved")
  parser.add_argument("--batch_size_bert",
                      default=32,
                      type=int,
                      help="entailment batch size")
  parser.add_argument("--batch_size_label",
                      default=24,
                      type=int,
                      help="entailment batch size")
  parser.add_argument("--main_dir",
                      default=None,
                      type=str,
                      help="Where source preprocess data are saved")
  parser.add_argument("--qnli_dir",
                      default=None,
                      type=str,
                      help="Where source preprocess qnli data are saved")
  parser.add_argument("--no_cuda",
                      action='store_true',
                      help="Whether not to use CUDA when available")
  parser.add_argument("--use_cuda",
                      action='store_true',
                      help="Whether not to use CUDA when available")
  parser.add_argument("--cache_dir",
                      default="",
                      type=str,
                      help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--lr_weight",
                      default=1e-4,
                      type=float,
                      help="Step size of weight A1 will be different than other params ?? could be... otherwise we observe almost 1/2 weights")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument('--fp16',
                      action='store_true',
                      help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument("--num_train_epochs_entailment",
                      default=5.0,
                      type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--num_train_epochs_bert",
                      default=1.0,
                      type=float,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                      "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--local_rank",
                      type=int,
                      default=-1,
                      help="local_rank for distributed training on gpus")
  parser.add_argument('--seed',
                      type=int,
                      default=2019,
                      help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps',
                      type=int,
                      default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--loss_scale',
                      type=float, default=0,
                      help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                      "0 (default value): dynamic loss scaling.\n"
                      "Positive power of 2: static loss scaling value.\n")
  parser.add_argument("--do_lower_case", action="store_true")
  parser.add_argument("--reduce_memory", action="store_true",
                      help="Store training data as on-disc memmaps to massively reduce memory usage")
  parser.add_argument('--pregenerated_data', type=Path, required=True)
  parser.add_argument('--bert_output_dir', type=Path, required=True)
  parser.add_argument('--write_vector', action='store_true',
                      help='write_vector of label emb')
  parser.add_argument('--label_desc_dir', type=str, default=None,
                      help='write_vector of label emb')

  args = parser.parse_args()
  return args
