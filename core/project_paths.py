from os import environ

if 'LOCAL_TEST' in environ:
    base_path = '/Users/leo101/Kaggle/rsna-2024-lumbar-spine-degenerative-classification/input/rsna-2024-lumbar-spine-degenerative-classification'
    save_path = '/Users/leo101/Kaggle/rsna-2024-lumbar-spine-degenerative-classification/checkpoints'
else:
    base_path = '/root/autodl-tmp/input/rsna-2024-lumbar-spine-degenerative-classification'
    save_path = '/root/autodl-fs'