import pandas as pd
import json

def parse_json_to_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    concepts = []
    frames = []
    descriptions = []
    interaction_types = []
    next_frames = []
    durations = []
    explanations = []

    #iterate over each entry in the JSON data
    for entry in data:
        concept = entry['concept']
        explanation = entry['explanation']
        asset_suggestions = entry['asset_suggestions']
        
        for asset in asset_suggestions:
            for frame, details in asset.items():
                concepts.append(concept)
                frames.append(frame)
                descriptions.append(details.get('Background', '') + ' ' + details.get('Foreground', '') + ' ' + details.get('Animation', '') + ' ' + details.get('Text', ''))
                interaction_types.append(details.get('Interactive Element', ''))
                next_frames.append(details.get('next_frame', ''))
                durations.append(details.get('duration', ''))
                explanations.append(explanation)

    # create a df
    df = pd.DataFrame({
        'Concept': concepts,
        'Frame': frames,
        'Description': descriptions,
        'Interaction Type': interaction_types,
        'Next Frame': next_frames,
        'Duration': durations,
        'Explanation': explanations
    })
    
    return df