import argparse
from transformers import (BertConfig, BertModel, BertTokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration, T5Tokenizer, BartConfig,
                          BartForConditionalGeneration, BartTokenizer, HoulsbyInvConfig, PfeifferConfig, ParallelConfig,
                          PrefixTuningConfig, LoRAConfig, IA3Config, AdapterConfig)
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
}

# roberta-base 125M
# microsoft/codebert-base 125M
# microsoft/unixcoder-base 126M
# facebook/bart-base 139M
# t5-base 223M
# t5-large 738M
# Salesforce/codet5-base 223M
# Salesforce/codet5-large 738M


def get_model_size(model, required=True):
    if required:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model_size = sum(p.numel() for p in model.parameters())
    return model_size / 1e+6


def getAdapter(adapter_type):
    if "houlsby" in adapter_type:
        adapter_config = HoulsbyInvConfig()
    elif "pfeiffer" in adapter_type:
        adapter_config = PfeifferConfig()
    elif "parallel" in adapter_type:
        adapter_config = ParallelConfig()
    elif "prefix_tuning" in adapter_type:
        adapter_config = PrefixTuningConfig()
    elif "lora" in adapter_type:
        adapter_config = LoRAConfig()
    elif "ia3" in adapter_type:
        adapter_config = IA3Config()
    else:
        adapter_config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor={'default': 16},
                                       non_linearity='gelu')
    return adapter_config


ADAPTER_TYPE = ["houlsby", "pfeiffer", "parallel", "prefix_tuning", "lora", "ia3"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--do_adapter", action='store_true', help="Whether to use adapter in model.")
    parser.add_argument('--adapter_name', type=str, default='search_adapter', help="Adapter name for each layer.")
    parser.add_argument("--adapter_type", type=str, default="houlsby",
                        choices=ADAPTER_TYPE, help="Adapter type to use.")
    parser.add_argument("--adapter_file", type=str, default=None,
                        help="Optional directory to store the pre-trained adapter.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")

    args = parser.parse_args()

    # build model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    logger.info('Used Model: {}'.format(args.model_name_or_path))
    # add adapter
    if args.do_adapter:
        adapter_config = getAdapter(args.adapter_type)

        if args.adapter_file:
            model.load_adapter(args.adapter_file)
        else:
            # task adapter - only add if not existing
            if args.adapter_name not in model.config.adapters:
                # add a new adapter
                model.add_adapter(args.adapter_name, config=adapter_config)
        # Enable adapter training
        model.train_adapter(args.adapter_name)
        model.set_active_adapters(args.adapter_name)

        logger.info('Used Adapter: {}'.format(args.adapter_type))

    num_param = get_model_size(model)
    num_total_param = get_model_size(model, required=False)
    percentage = (num_param / num_total_param) * 100
    logger.info(
        'Number of total parameters: {} M, tunable parameters: {} M, percentage: {}'.format(num_total_param, num_param,
                                                                                            percentage))
    logger.info("***********************************************************")


if __name__ == "__main__":
    main()
