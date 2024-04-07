from typing import List, Protocol, TypeVar, Union
# from langchain.schema.embeddings import Embeddings
import requests
import json
import time
from chromadb import Embeddings
from datetime import datetime, timedelta
from tqdm.notebook import tqdm
import together

import logging
logger = logging.getLogger(__name__)


class CustomEmbeddings(Embeddings):
    url = 'http://158.160.50.98:8002/embeddings'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    def __call__(self, texts) -> Embeddings:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(self.url, headers=self.headers, data=json.dumps([text]))
        embeddings = response.json()['embeddings']
        return embeddings[0]

    def embed_documents(self, texts):
        response = requests.post(self.url, headers=self.headers, data=json.dumps(texts))
        embeddings = response.json()['embeddings']
        return embeddings

    
class TogetherEmbeddings(Embeddings):
    def __init__(self, model, token):
        self.client = together.Together(api_key=token)
        self.model = model
        
    def __call__(self, texts: List[str]) -> Embeddings:
        return self.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.embed_documents([text])[0]
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [text.replace("\n", " ") for text in texts]
        outputs = []
        for text in texts:
            outputs.append(self.client.embeddings.create(input = [text], model=self.model).data[0].embedding)
            time.sleep(0.5)
        return outputs
    

class YandexEmbeddings(Embeddings):  
    def __init__(self, folder_id, oauth_token):
        self.folder_id = folder_id
        self.oauth_token = oauth_token
        
        self._iam_expires_at = None
        self._iam_token = None
        
        self._update_iam_token()
        
        self._embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        self._doc_uri = f"emb://{folder_id}/text-search-doc/latest"
        self._query_uri = f"emb://{folder_id}/text-search-query/latest"

    
    def _update_iam_token(self):
        logger.info('Updating Yandex IAM token.')
        query = {
            "yandexPassportOauthToken": self.oauth_token
        }
        response = requests.post("https://iam.api.cloud.yandex.net/iam/v1/tokens", data=json.dumps(query))
        if response.status_code != 200:
           logger.error(f"Could not update Yandex IAM token. Status code: {response.status_code} Body:\n{response.text}")
           raise Exception('Failed to get Yandex IAM token!')

        response = response.json()
        self._iam_expires_at = datetime.strptime(response['expiresAt'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
        self._iam_token = response['iamToken']
        logger.info(f'IAM token succesully updated: {self._iam_token[:15]} Expires at: {self._iam_expires_at}')
    
    
    def __call__(self, texts: List[str]) -> Embeddings:
        return self.embed_documents(texts)
    
    
    def embed_query(self, query: str) -> List[float]:
        if datetime.utcnow() + timedelta(hours=1) > self._iam_expires_at:
            self._update_iam_token()
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._iam_token}",
            "x-folder-id": f"{self.folder_id}"
        }

        body = {
            "modelUri": self._query_uri,
            "text": query,
        }
        
        response = requests.post(self._embed_url, json=body, headers=headers)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise Exception(f'Failed to get embedding. Status code: {response.status_code}')
        
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._iam_token}",
                "x-folder-id": f"{self.folder_id}"
            }
        
        embeddings = []
        
        for text in tqdm(texts):
            if datetime.utcnow() + timedelta(hours=1) > self._iam_expires_at:
                self._update_iam_token()
                headers['Authorization'] = f"Bearer {self._iam_token}"

            body = {
                "modelUri": self._doc_uri,
                "text": text,
            }

            retries = 0
            while retries < 5:
                response = requests.post(self._embed_url, json=body, headers=headers)
                
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                    break
                    
                elif response.status_code == 429:
                    time.sleep(1)
                    retries += 1
                else:
                    raise Exception(f'Failed to get embedding. Status code: {response.status_code}\nMessage:{response.text}')
        return embeddings
    