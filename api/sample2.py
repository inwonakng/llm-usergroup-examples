import os
import requests 
from chat_api import DEFAULT_CHAT_PARAMS
import html

instruction = 'Your job is to play the assigned role and give responses to your best ability.\n'
chat_history = [
    [
        'You are a helpful assistant. You will answer questions I ask you. Reply with Yes if you understand.',
        'Yes, I understand'
    ]
]
params = dict(
    **DEFAULT_CHAT_PARAMS,
    user_input = 'What color is the sky?',
    history = dict(
        internal = chat_history,
        visible = chat_history,
    ),
    context_instruct = instruction,
)
response = requests.post('http://localhost:5000/api/v1/chat', json=params)
result = response.json()['results'][0]['history']
output = html.unescape(result['visible'][-1][1])
print(output)