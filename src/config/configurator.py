import os
import yaml
import argparse
import torch

def update_configs(configs, up):
    for k, v in up.items():
        if k not in configs:
            configs[k] = v
        elif isinstance(v, dict):
            update_configs(configs[k], v)
        else:
            configs[k] = v

def parse_configure():
    parser = argparse.ArgumentParser(description='SSLRec')
    parser.add_argument('--model', type=str, default="daksr", help='Model name') 
    parser.add_argument('--config_name', type=str, default=None, help='Specify the config name')
    parser.add_argument('--config_list', type=str, default=None, help='Config list')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Dataset name')
    parser.add_argument('--logname', type=str, default=None, help='Log name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune'], help='Device number')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        if args.cuda == 'all':
            args.cuda = ','.join([str(i) for i in range(torch.cuda.device_count())])
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")


    yml_fn = './config/modelconf/{}.yml'.format(model_name if args.config_name is None else args.config_name)
    with open(yml_fn, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)

        # model name
        configs['model']['name'] = configs['model']['name'].lower()
        
        # log dir
        configs['model']['logname'] = configs['model']['name'] if (not args.logname) else args.logname

        # grid search
        if 'tune' not in configs or args.mode!='tune' :
            configs['tune'] = {'enable': False}

        # gpu device
        configs['device'] = args.device

        # dataset
        if args.dataset is not None:
            configs['data']['name'] = args.dataset

        # log
        if 'log_loss' not in configs['train']:
            configs['train']['log_loss'] = True

        # early stop
        if 'patience' in configs['train']:
            if configs['train']['patience'] <= 0:
                raise Exception("'patience' should be greater than 0.")
            else:
                configs['train']['early_stop'] = True
        else:
            configs['train']['early_stop'] = False

    # for overwrite configs
    if args.config_list:
        config_list = args.config_list.split(',')
        for config_name in config_list:
            fn = f"./config/override/{config_name}.yml"
            if not os.path.exists(fn):
                raise Exception(f"Config file {fn} does not exist.")
            with open(fn, encoding='utf-8') as f:
                config_data = f.read()
                config = yaml.safe_load(config_data)
                update_configs(configs, config)

    return configs

configs = parse_configure()
