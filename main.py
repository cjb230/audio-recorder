# main.py
import pyaudio
import wave
import soundfile as sf
import numpy as np
import webrtcvad
import logging
from datetime import datetime
import os
import config


# Suppress unnecessary log messages
logging.basicConfig(level=logging.WARNING)

def suppress_pyaudio_logs():
    """
    Suppresses PyAudio and webrtcvad logs to reduce console noise.
    """
    logging.getLogger("pyaudio").setLevel(logging.ERROR)
    logging.getLogger("webrtcvad").setLevel(logging.ERROR)


class AudioRecorder:
    """
    Records audio from a Blue Snowball microphone, detects speech via WebRTC VAD,
    and saves audio files to either a network drive or a backup folder upon silence.
    """

    def __init__(self):
        suppress_pyaudio_logs()
        self.sample_rate = config.SAMPLE_RATE
        self.frame_duration = config.FRAME_DURATION  # in milliseconds
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.vad = webrtcvad.Vad(config.VAD_SENSITIVITY)
        self.recording = False
        self.buffer = []

        # Number of silent frames allowed before saving
        self.silence_threshold = int(
            (config.SILENCE_DURATION_BEFORE_SAVE * 1000) / self.frame_duration
        )
        self.silence_counter = 0

        # Number of silence frames to keep at the end of the recording
        self.trailing_silence_frames = int(
            (config.TRAILING_SILENCE_TO_KEEP * 1000) / self.frame_duration
        )

        # Paths
        self.network_path = config.NETWORK_PATH
        self.backup_path = config.BACKUP_PATH

        if not os.path.exists(self.backup_path):
            os.makedirs(self.backup_path)

        # Flag to indicate if any speech was detected during a recording session
        self.speech_detected_in_recording = False

    def _get_blue_snowball_device_index(self, audio_obj: pyaudio.PyAudio) -> int:
        """
        Finds and returns the device index of the 'Blue Snowball' microphone.
        Raises an exception if not found.
        """
        for i in range(audio_obj.get_device_count()):
            device_info = audio_obj.get_device_info_by_index(i)
            if "snowball" in device_info["name"].lower():
                return i
        raise Exception("Blue Snowball microphone not found")

    def record(self):
        """
        Continuously reads audio frames from the microphone and detects speech.
        Once 5 consecutive speech frames are detected, recording starts.
        Pre-buffer ensures initial speech frames are included.
        """
        audio = pyaudio.PyAudio()
        try:
            device_index = self._get_blue_snowball_device_index(audio)
        except Exception as e:
            audio.terminate()
            raise e

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.frame_size
        )

        print("Listening for speech... Press Ctrl+C to stop.")

        self.speech_frame_counter = 0  # Track consecutive speech frames
        self.speech_pre_buffer = []  # Store speech frames before recording starts

        try:
            while True:
                frame = stream.read(self.frame_size, exception_on_overflow=False)

                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                except Exception:
                    pass  # If VAD fails, treat as non-speech

                if is_speech:
                    self.speech_detected_in_recording = True
                    self.silence_counter = 0
                    self.speech_frame_counter += 1  # Count consecutive speech frames

                    # Always store speech frames in the pre-buffer
                    self.speech_pre_buffer.append(frame)

                    # Only start recording if we get 5 consecutive speech frames
                    if not self.recording and self.speech_frame_counter >= config.MINIMUM_CONSECUTIVE_SPEECH_FRAMES:
                        print("Speech confirmed, recording started")
                        self.recording = True
                        self.buffer = self.speech_pre_buffer[:]  # Include pre-buffer
                        self.speech_pre_buffer = []  # Clear pre-buffer

                    if self.recording:
                        self.buffer.append(frame)

                else:  # No speech detected
                    self.speech_frame_counter = 0  # Reset speech frame counter
                    if not self.recording:
                        self.speech_pre_buffer = []  # Discard old pre-buffer on silence

                    if self.recording:
                        self.silence_counter += 1
                        self.buffer.append(frame)

                        # Stop recording after prolonged silence
                        if self.silence_counter >= self.silence_threshold:
                            if self.speech_detected_in_recording:
                                self.save_recording()
                            self.recording = False
                            self.buffer = []
                            self.silence_counter = 0
                            self.speech_detected_in_recording = False

        except KeyboardInterrupt:
            print("Recording stopped by user")
            if self.recording and self.buffer and self.speech_detected_in_recording:
                self.save_recording()
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def save_recording(self):
        """
        Saves the current buffer of audio frames to disk, first trying
        the network path, then falling back to the backup path if there's an error.
        """
        if len(self.buffer) <= 1:
            return

        # Trim trailing silence if needed
        if self.silence_counter > self.trailing_silence_frames:
            trim_frames = self.silence_counter - self.trailing_silence_frames
            self.buffer = self.buffer[:-trim_frames]

        # Construct filename
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{config.OUTPUT_FORMAT}"
        audio_data = np.frombuffer(b''.join(self.buffer), dtype=np.int16)

        # Attempt to save to the network drive
        network_file_path = os.path.join(self.network_path, filename)
        try:
            if not os.path.exists(self.network_path):
                raise FileNotFoundError(
                    f"Network directory {self.network_path} does not exist"
                )
            self._write_audio_file(audio_data, network_file_path)
            print(f"Saved recording to network drive: {network_file_path}")

        # Fallback to the backup path
        except Exception:
            backup_file_path = os.path.join(self.backup_path, filename)
            try:
                self._write_audio_file(audio_data, backup_file_path)
                print(f"Saved recording to backup location: {backup_file_path}")
            except Exception as e:
                print(f"Failed to save recording: {str(e)}")

    def _write_audio_file(self, audio_data: np.ndarray, filepath: str):
        """
        Writes the audio data to a file in the format specified by config.OUTPUT_FORMAT.
        """
        if config.OUTPUT_FORMAT.lower() == "wav":
            self._save_to_wave_file(audio_data, filepath)
        else:
            self._save_to_flac_file(audio_data, filepath)

    def _save_to_wave_file(self, audio_data: np.ndarray, filepath: str):
        """
        Saves audio data in WAV format.
        """
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def _save_to_flac_file(self, audio_data: np.ndarray, filepath: str):
        """
        Saves audio data in FLAC format using the soundfile library.
        """
        with sf.SoundFile(
            filepath, mode='x', samplerate=self.sample_rate,
            channels=1, subtype='PCM_16', format="FLAC"
        ) as sf_file:
            sf_file.write(audio_data)


if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.record()
