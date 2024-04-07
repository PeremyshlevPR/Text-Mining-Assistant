import requests
import json
import time
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

from prompt_templates import QUERY_WITH_CONTEXT_TEMPLATE, DEFAULT_SYSTEM_PROMPT
from conf import settings

class YandexGPT:
    def __init__(self,
                 folder_id,
                 oauth_token,
                 model_version: str | int = "latest",
                 system_prompt: int = DEFAULT_SYSTEM_PROMPT
                 ):
        self.folder_id = folder_id
        self.oauth_token = oauth_token
        
        self._iam_expires_at = None
        self._iam_token = None  
        self._update_iam_token()

        self._model_uri = f'gpt://{self.folder_id}/yandexgpt/{model_version}'
        self.system_prompt = system_prompt

        self._completion_uri = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._max_retries = 5

    def _update_iam_token(self):
        logger.debug('Updating Yandex IAM token.')
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
        logger.debug(f'IAM token succesully updated: {self._iam_token[:15]} Expires at: {self._iam_expires_at}')

    @staticmethod
    def _get_model_input(prompt, context):
        if not context:
            return prompt
        return QUERY_WITH_CONTEXT_TEMPLATE.format(question=prompt, context=context)


    def __call__(self,
                 prompt: str,
                 context: str | None = None,
                 temperature: float = 0.5,
                 max_new_tokens: int = 1024
                ):
        if datetime.utcnow() + timedelta(hours=1) > self._iam_expires_at:
                self._update_iam_token()

        headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._iam_token}",
                "x-folder-id": f"{self.folder_id}"
            }
        
        body = {
          "modelUri": self._model_uri,
          "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_new_tokens
          },
          "messages": [
            {
              "role": "system",
              "text": self.system_prompt
            },
            {
              "role": "user",
              "text": self._get_model_input(prompt=prompt, context=context)
            }
          ]
        }
        logger.debug(f'Got query to YandexGPT:\n{body["messages"][-1]["text"]}')

        retries = 0
        while retries < self._max_retries:   
            response = requests.post(self._completion_uri, headers=headers, data=json.dumps(body))
            
            if response.status_code == 200:
                logger.info('LLM succesfully generated answer')
                return response.json()['result']['alternatives'][0]['message']['text']
            
            elif response.status_code == 429:
                time.sleep(1)
            
            retries += 1
            logger.error(f'Generation failed! Retires {retries}/{self._max_retries}. Status code: {response.status_code}\n{response.text}')
        raise Exception(f'Generation failed! Status code: {response.status_code}\n{response.text}')
