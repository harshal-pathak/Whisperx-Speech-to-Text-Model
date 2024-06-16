import whisperx
import gc 

device = "cpu" 
audio_file = "" #Add mp3 file here to transcribe
batch_size = 16 
compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy) #Mostly for cpu you can go with float32


model = whisperx.load_model("base", device, compute_type=compute_type) #base model is quite good for result

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) 

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"])

diarize_model = whisperx.DiarizationPipeline(use_auth_token="ADD YOUR HF Token with write rights", device=device)

diarize_segments=diarize_model(audio, min_speakers=2, max_speakers=5) #Min and Max Speaker is optional 

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs 
