from decouple import config

API_KEY = config('API_KEY')
MODEL = config('MODEL', default='gpt-4')
API_BASE = config('API_BASE', default='https://api.openai.com/v1')
API_TIMEOUT = config('API_TIMEOUT', default=30, cast=int)