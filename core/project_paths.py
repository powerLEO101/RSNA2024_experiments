from os import environ

if 'LOCAL_TEST' in environ:
    base_path = '/Users/leo101/Kaggle/rsna-2024-lumbar-spine-degenerative-classification/input/rsna-2024-lumbar-spine-degenerative-classification'
    save_path = '/Users/leo101/Kaggle/rsna-2024-lumbar-spine-degenerative-classification/input/RSNA24_checkpoints'
else:
    base_path = '/media/workspace/RSNA2024_input/rsna-2024-lumbar-spine-degenerative-classification'
    save_path = '/media/workspace/RSNA2024_checkpoints'