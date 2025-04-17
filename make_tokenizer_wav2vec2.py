from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

model_path = "/home/kmjones/.cache/huggingface/hub/models--facebook--wav2vec2-large-xlsr-53-dutch/snapshots/ced00f8603b017d58cb3bcb3883e8c28f19ccdb4"
print("starting...")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
print("tokenizer loaded")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
print("feature extractor loaded")


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Load tokenizer from vocab.json and other files
print("processor loaded")
# Save the tokenizer again — this will create tokenizer.json
processor.save_pretrained(model_path)
print("tokenizer saved")

