import requests
from datetime import datetime, timedelta
import json
from typing import List
import time


class YandexTranslator:  
    def __init__(self, folder_id, oauth_token):
        self.folder_id = folder_id
        self.oauth_token = oauth_token
        
        self._iam_expires_at = None
        self._iam_token = None
        
        self._update_iam_token()
        
        self._url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        
        self.batch_size = 5

    
    def _update_iam_token(self):
        print('Updating Yandex IAM token.')
        query = {
            "yandexPassportOauthToken": self.oauth_token
        }
        response = requests.post("https://iam.api.cloud.yandex.net/iam/v1/tokens", data=json.dumps(query))
        if response.status_code != 200:
            print(f"Could not update Yandex IAM token. Status code: {response.status_code} Body:\n{response.text}")
            raise Exception('Failed to get Yandex IAM token!')

        response = response.json()
        self._iam_expires_at = datetime.strptime(response['expiresAt'].split('.')[0], "%Y-%m-%dT%H:%M:%S")
        self._iam_token = response['iamToken']
        print(f'IAM token succesully updated: {self._iam_token[:15]} Expires at: {self._iam_expires_at}')
    
    
    def __call__(self, texts: List[str], source='en', target='ru') -> List[str]:
        return self.translate(texts=texts, source=source, target=target)
    
    
    def translate(self, texts: List[str], source='en', target='ru') -> List[str]:
        if isinstance(texts, str):
            texts = [texts]
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._iam_token}",
            "x-folder-id": f"{self.folder_id}"
        }
        
        translated_texts = []
        
        for i in range(0, len(texts), self.batch_size):
            if datetime.utcnow() + timedelta(hours=1) > self._iam_expires_at:
                self._update_iam_token()
                headers['Authorization'] = f"Bearer {self._iam_token}"
                
            body = {
                "folderId": self.folder_id,
                "texts": texts[i: i + self.batch_size],
                "targetLanguageCode": target
            }

            retries = 0
            while retries < 5:
                try:
                    response = requests.post(self._url, json=body, headers=headers)
                    
                    if response.status_code == 200:
                        translated_texts += [translation['text'] for translation in response.json()['translations']]
                        break
                    else:
                        time.sleep(1)
                        retries += 1
                except Exception as e:
                    print(e)
                    print(response.text)
            else:
                raise Exception(f'Failed to translate text. Status code: {response.status_code}\nMessage:{response.text}')
        return translated_texts