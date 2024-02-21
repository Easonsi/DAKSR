from config.configurator import configs
import importlib

def build_trainer(data_handler, logger):
    trainer_name = 'Trainer' if 'trainer' not in configs['train'] else configs['train']['trainer']
    trainer_name = trainer_name.replace('_', '')
    trainers = importlib.import_module('trainer.trainer')
    for attr in dir(trainers):
        if attr.lower() == trainer_name.lower():
            return getattr(trainers, attr)(data_handler, logger)
    # trainers2 = importlib.import_module('trainer.trainer2')
    # for attr in dir(trainers2):
    #     if attr.lower() == trainer_name.lower():
    #         return getattr(trainers2, attr)(data_handler, logger)
    trainers3 = importlib.import_module('trainer.trainer3')
    for attr in dir(trainers3):
        if attr.lower() == trainer_name.lower():
            return getattr(trainers3, attr)(data_handler, logger)
    else:
        raise NotImplementedError('Trainer Class {} is not defined in {}'.format(trainer_name, 'trainer.trainer'))
