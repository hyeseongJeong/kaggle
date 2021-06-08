import yaml
import argparse
from dogs_vs_cats.src.pipe_line_handler import PipeLineHandler


def get_trainee_info_dict(config_path):
    with open(config_path, 'r') as f:
        trainee_info = yaml.load(f, Loader=yaml.FullLoader)
    return trainee_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='python --config_path /path/to/model_config.yaml')
    args = parser.parse_args()

    trainee_info_dict = get_trainee_info_dict(args.config_path)

    # config_path = '/dogs_vs_cats/src/trainee_info.yml'
    # trainee_info_dict = get_trainee_info_dict(config_path)
    pipe_line_handler = PipeLineHandler(trainee_info_dict)
    pipe_line_handler.run()


