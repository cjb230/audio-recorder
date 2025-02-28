# Output format: 'flac' or 'wav'
OUTPUT_FORMAT = "flac"

# Storage paths
NETWORK_PATH = "/mnt/smbhome/audio_in/"
BACKUP_PATH = "./backup_audio/"

# Audio settings
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # in milliseconds
SILENCE_DURATION_BEFORE_SAVE = 3  # seconds
TRAILING_SILENCE_TO_KEEP = 0.1  # seconds
VAD_SENSITIVITY = 3  # set aggressiveness mode. 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
MINIMUM_CONSECUTIVE_SPEECH_FRAMES = 7
