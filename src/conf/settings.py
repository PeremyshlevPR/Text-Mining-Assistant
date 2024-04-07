from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ROOT_LOG_LEVEL: str = "INFO"

    YANDEX_OAUTH_TOKEN: str
    YANDEX_FOLDER_ID: str
    VECTORSTORE_DIR: str

    TOGETHER_TOKEN: str
    EMBEDDING_MODEL: str

    model_config = SettingsConfigDict(
        env_file='conf/.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='allow' 
    )

settings = Settings.model_validate({})
