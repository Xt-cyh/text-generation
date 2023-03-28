from googleapiclient import discovery
import json


def detect_toxic(text):

    # Replace with your own Perspective API key
    # https://console.cloud.google.com/apis/credentials?project=detoxic-381613
    API_KEY = 'AIzaSyBI4uQ4-8PAprclz_vaFEqetUSvjFa00ok'

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {
            'TOXICITY': {}
        }
    }

    return client.comments().analyze(body=analyze_request).execute()

response = detect_toxic('you are welcome')
print(json.dumps(response, indent=2)) 
