from googleapiclient import discovery
import pandas as pd
from tqdm import tqdm
import json
import time
import os


# 设置代理环境变量
os.environ["http_proxy"] = "http://127.0.0.1:47890"
os.environ["https_proxy"] = "http://127.0.0.1:47890"


def detect_toxic(texts):
    
    # Replace with your own Perspective API key
    # https://console.cloud.google.com/apis/credentials?project=detoxic-381613
    API_KEY = 'AIzaSyCUtj8D8keMoZpEJKPBM6EoKIi6WMqv6uU'

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    toxicity = []
    for i in tqdm(range(len(texts)), desc='Toxicity:'):
        analyze_request = {
            'comment': {'text': texts[i]},
            'requestedAttributes': {
                'TOXICITY': {}
                # 'SEVERE_TOXICITY':{}
            }
        }
        response = client.comments().analyze(body=analyze_request).execute()
        ''' response structure:
        {
            'attributeScores': {
                'TOXICITY': {
                    'spanScores': [{'begin': 0, 'end': 16, 'score': {'value': 0.02048268, 'type': 'PROBABILITY'}}], 
                    'summaryScore': {'value': 0.02048268, 'type': 'PROBABILITY'}
                }, 
                'SEVERE_TOXICITY': {
                    'spanScores': [{'begin': 0, 'end': 16, 'score': {'value': 0.0014877319, 'type': 'PROBABILITY'}}], 
                    'summaryScore': {'value': 0.0014877319, 'type': 'PROBABILITY'}
                }
            }, 
            'languages': ['en'], 
            'detectedLanguages': ['en']
        }
        '''
        toxicity.append(response['attributeScores']['TOXICITY']['summaryScore']['value'])
        # time.sleep(1)
    return toxicity


if __name__ == "__main__":
    texts = [
        'you are welcome.',
        'you are so stupid!'
    ]
    response = detect_toxic(texts)
    print(response)
