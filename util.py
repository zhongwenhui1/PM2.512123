import yaml
import sys
import os
import numpy as np


proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# Universal file path configuration - works on any server
try:
    nodename = os.uname().nodename
except AttributeError:
    # Windows compatibility
    import platform
    nodename = platform.node()

# Try to get server-specific path, fallback to default
if 'filepath' in config and nodename in config['filepath']:
    file_dir = config['filepath'][nodename]
elif 'GPU-Server' in config.get('filepath', {}):
    # Fallback to GPU-Server configuration
    file_dir = config['filepath']['GPU-Server']
else:
    # Final fallback - use default paths
    file_dir = {
        'knowair_fp': './processed_data/processed_data.npy',
        'results_dir': './results'
    }


def main():
    pass


if __name__ == '__main__':
    main()
