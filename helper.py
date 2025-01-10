import os
import warnings
import logging

def setup_environment():
    warnings.filterwarnings('ignore')
    logging.getLogger("torch").setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

def print_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                             Proposed Name: PixEye                         ║
    ║                             SAFETYWHAT ASSESSMENT                         ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print("\033[95m" + banner + "\033[0m") 