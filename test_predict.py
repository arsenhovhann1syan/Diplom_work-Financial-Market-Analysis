import json
from src.inference.predict import predict_single

with open("models/features.json", "r") as f:
    features = json.load(f)

# dummy input — բոլոր feature-ներին տալիս ենք 0
sample_input = {
    feature: 0.0
    for feature in features
}

result = predict_single(sample_input)

print(result)