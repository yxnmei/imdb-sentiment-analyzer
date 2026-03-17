from huggingface_hub import login, HfApi
import time

token = input("Enter HuggingFace token: ")
login(token=token)

api = HfApi()
api.create_repo(repo_id='yxnmei/imdb-lstm-sentiment', exist_ok=True)

for attempt in range(3):
    try:
        api.upload_file(path_or_fileobj='notebooks/models/lstm_best.pt', path_in_repo='lstm_best.pt', repo_id='yxnmei/imdb-lstm-sentiment')
        api.upload_file(path_or_fileobj='notebooks/models/vocab.json',   path_in_repo='vocab.json',   repo_id='yxnmei/imdb-lstm-sentiment')
        print('✅ Done! https://huggingface.co/yxnmei/imdb-lstm-sentiment')
        break
    except Exception as e:
        print(f'Attempt {attempt+1} failed: {e}. Retrying in 5s...')
        time.sleep(5)