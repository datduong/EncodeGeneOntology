
from argparse import ArgumentParser
from pathlib import Path

# start_near_true_label = False ## global


def get_args():
  parser = ArgumentParser(
      description='encode label descriptions into vectors')

  parser.add_argument('--word_mode', type=str, default=None,
                      help='do we use bilstm or avepool')
  parser.add_argument('--lr', type=float, default=0.0001,
                      help='learning rate')
  parser.add_argument('--result_folder', type=str, default=None,
                      help='where to save result')
  parser.add_argument('--epoch', type=int, default=200,
                      help='num of iteration')
  parser.add_argument('--def_emb_dim', type=int, default=300,
                      help='emb dim for definition of GO terms')
  parser.add_argument('--vector_file', type=str, default=None,
                      help='file of vectors')
  parser.add_argument('--word_emb_dim', type=int, default=300,
                      help='emb dim for word')
  parser.add_argument('--model_load', type=str, default=None,
                      help='path to a model to load')
  parser.add_argument('--w2v_emb', type=str, default=None,
                      help='path to a w2v emb')
  parser.add_argument('--vocab_list', type=str, default=None,
                      help='path to a vocab')
  parser.add_argument('--not_train_w2v_emb', action='store_true',
                      help='do not train word emb')
  parser.add_argument('--do_test', action='store_true',
                      help='do_test only')
  parser.add_argument('--metric_option', type=str,
                      help='cosine or entailment')
  parser.add_argument('--test_file', type=str, default=None,
                      help='where test file at')
  parser.add_argument('--write_score', type=str, default=None,
                      help='write out score')
  parser.add_argument('--nonlinear_gcnn', type=str, default=None,
                      help='tanh relu or something? should be tanh if use cosine metric')
  parser.add_argument('--do_gcnn', action='store_true',
                      help='run graph cnn')
  parser.add_argument('--gcnn_dim', type=int, default=300,
                      help='emb dim for the gcnn vec')
  parser.add_argument('--bilstm_dim', type=int, default=300,
                      help='emb dim for the bilstm vec')
  parser.add_argument('--pretrained_label_emb', type=str, default=None,
                      help='pretrained_label_emb for gcnn')
  parser.add_argument('--do_continue', action='store_true',
                      help='continue training only')
  parser.add_argument('--reduce_cls_vec', action='store_true',
                      help='768 to smaller dim then compare the cosine')
  parser.add_argument('--update_bert', action='store_true',
                      help='update bert model, do this rarely')
  parser.add_argument('--fix_word_emb', action='store_true',
                      help='do not update w2v or glove emb')
  parser.add_argument('--num_label', type=int, default=0,
                      help='how many nodes')
  parser.add_argument('--batch_size_label_desc', type=int, default=0,
                      help='must pass label through lstm or something then use smaller batch')
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
                      default=42,
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
  parser.add_argument('--pregenerated_data', type=Path)
  parser.add_argument('--bert_output_dir', type=Path)
  parser.add_argument('--write_vector', action='store_true',
                      help='write_vector of label emb')
  parser.add_argument('--label_desc_dir', type=str, default=None,
                      help='write_vector of label emb')

  args = parser.parse_args()
  return args
