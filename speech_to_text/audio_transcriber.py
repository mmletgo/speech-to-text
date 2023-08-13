import asyncio
import functools
import eel
import queue
import numpy as np

import traceback
import requests
import os
import copy
from datetime import timedelta
from notion_client import Client
from scipy.io import wavfile
import io

from typing import NamedTuple
# from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor

from .utils.audio_utils import create_audio_stream
from .vad import Vad
from .utils.file_utils import write_audio
from .websoket_server import WebSocketServer
from .openai_api import OpenAIAPI


class AppOptions(NamedTuple):
    audio_device: int
    silence_limit: int = 8
    noise_threshold: int = 5
    non_speech_threshold: float = 0.1
    include_non_speech: bool = False
    create_audio_file: bool = True
    use_websocket_server: bool = False
    use_openai_api: bool = False

    notion_apikey: str = ''
    courselist_databaseid: str = ''
    course: str = ''
    note_name: str = 'note'
    myserver: str = ''


class AudioTranscriber:

    def __init__(
        self,
        event_loop: asyncio.AbstractEventLoop,
        whisper_model,
        transcribe_settings: dict,
        app_options: AppOptions,
        websocket_server: WebSocketServer,
        openai_api: OpenAIAPI,
        coursedict: dict,
    ):
        self.event_loop = event_loop
        self.whisper_model = whisper_model
        self.transcribe_settings = transcribe_settings
        self.app_options = app_options
        self.websocket_server = websocket_server
        self.openai_api = openai_api
        self.vad = Vad(app_options.non_speech_threshold)
        self.silence_counter: int = 0
        self.audio_data_list = []
        self.all_audio_data_list = []
        self.audio_queue = queue.Queue()
        self.transcribing = False
        self.stream = None
        self._running = asyncio.Event()
        self._transcribe_task = None

        if self.app_options.notion_apikey != '':
            self.notion = Client(auth=self.app_options.notion_apikey)
            self.coursedict = coursedict

    def mytranscribe(self, audio_data):
        # with open("test_audio0.wav", "wb") as audio_file:
        #     audio_file.write(audio)
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 16000, audio_data)
        # audio_bytes = bytes(audio_data)
        wav_buffer.seek(0)
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        response = requests.post('http://' + self.app_options.myserver +
                                 '/asr',
                                 files=files)

        if response.status_code == 200:
            result = response.json()
            segment_list = result["transcription"]
            return segment_list, None
        else:
            print("Error:", response.text)

    async def transcribe_audio(self):
        # Ignore parameters that affect performance
        transcribe_settings = self.transcribe_settings.copy()
        transcribe_settings["without_timestamps"] = True
        transcribe_settings["word_timestamps"] = False

        with ThreadPoolExecutor() as executor:
            while self.transcribing:
                try:
                    # Get audio data from queue with a timeout
                    audio_data = await self.event_loop.run_in_executor(
                        executor,
                        functools.partial(self.audio_queue.get, timeout=3.0))
                    if self.app_options.myserver != '':
                        # Create a partial function for the model's transcribe method
                        func = functools.partial(self.mytranscribe,
                                                 audio_data=audio_data)

                        # Run the transcribe method in a thread
                        segments, _ = await self.event_loop.run_in_executor(
                            executor, func)

                        for segment in segments:
                            eel.display_transcription(segment['text'])
                            if self.websocket_server is not None:
                                await self.websocket_server.send_message(
                                    segment['text'])
                    else:
                        # Create a partial function for the model's transcribe method
                        func = functools.partial(
                            self.whisper_model.transcribe,
                            audio=audio_data,
                            **transcribe_settings,
                        )

                        # Run the transcribe method in a thread
                        segments, _ = await self.event_loop.run_in_executor(
                            executor, func)

                        for segment in segments:
                            eel.display_transcription(segment.text)
                            if self.websocket_server is not None:
                                await self.websocket_server.send_message(
                                    segment.text)

                except queue.Empty:
                    # Skip to the next iteration if a timeout occurs
                    continue
                except Exception:
                    eel.on_recive_message(str(traceback.format_exc()))

    def process_audio(self, audio_data: np.ndarray, frames: int, time, status):
        is_speech = self.vad.is_speech(audio_data)
        if is_speech:
            self.silence_counter = 0
            self.audio_data_list.append(audio_data.flatten())
        else:
            self.silence_counter += 1
            if self.app_options.include_non_speech:
                self.audio_data_list.append(audio_data.flatten())

        if not is_speech and self.silence_counter > self.app_options.silence_limit:
            self.silence_counter = 0

            if self.app_options.create_audio_file:
                self.all_audio_data_list.extend(self.audio_data_list)

            if len(self.audio_data_list) > self.app_options.noise_threshold:
                concatenate_audio_data = np.concatenate(self.audio_data_list)
                self.audio_data_list.clear()
                self.audio_queue.put(concatenate_audio_data)
            else:
                # noise clear
                self.audio_data_list.clear()

    def batch_transcribe_audio(self, audio_data: np.ndarray):
        if self.app_options.myserver != '':
            segment_list, _ = self.mytranscribe(audio_data)
        else:
            segment_list = []
            segments, _ = self.whisper_model.transcribe(
                audio=audio_data, **self.transcribe_settings)

            for segment in segments:
                word_list = []
                if self.transcribe_settings["word_timestamps"] is True:
                    for word in segment.words:
                        word_list.append({
                            "start": word.start,
                            "end": word.end,
                            "text": word.word,
                        })
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": word_list,
                })

        eel.transcription_clear()

        if self.openai_api is not None:
            self.text_proofreading(segment_list)
        else:
            eel.on_recive_segments(segment_list)

        try:
            srt_filename = 'web/record/' + self.app_options.note_name + ".srt"
            if self.app_options.course != '':
                srt_filename = 'web/record/' + self.app_options.course + '/' + self.app_options.note_name + ".srt"
            self.generate_srt_file(segment_list, srt_filename)
            if self.app_options.notion_apikey != '':
                if self.app_options.course in self.coursedict.keys():
                    database_id = self.coursedict[self.app_options.course]
                    self.save_segment_list_to_notion(segment_list, database_id)
        except Exception:
            eel.on_recive_message(str(traceback.format_exc()))

    def text_proofreading(self, segment_list: list):
        # Use [#] as a separator
        combined_text = "[#]" + "[#]".join(segment["text"]
                                           for segment in segment_list)
        result = self.openai_api.text_proofreading(combined_text)
        split_text = result.split("[#]")

        del split_text[0]

        eel.display_transcription("Before text proofreading.")
        eel.on_recive_segments(segment_list)

        if len(split_text) == len(segment_list):
            for i, segment in enumerate(segment_list):
                segment["text"] = split_text[i]
                segment["words"] = []
            eel.on_recive_message("proofread success.")
            eel.display_transcription("After text proofreading.")
            eel.on_recive_segments(segment_list)
        else:
            eel.on_recive_message("proofread failure.")
            eel.on_recive_message(result)

    def generate_srt_file(self, segment_list, filename):
        with open(filename, 'w') as f:
            for i, segment in enumerate(segment_list):
                start_time = timedelta(seconds=segment['start'])
                end_time = timedelta(seconds=segment['end'])
                subtitle = segment['text']

                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{subtitle}\n\n")

    def save_segment_list_to_notion(self, segment_list, database_id):
        linetext = ''
        alltext = ''
        new_block = []

        for segment in segment_list:
            subtitle = segment['text'] + '\n'
            templine = linetext + subtitle
            if len(templine) > 2000:
                new_block.append({
                    "object": "block",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": copy.deepcopy(linetext),
                            },
                        }],
                    }
                })
                linetext = subtitle
            else:
                linetext += subtitle

        if linetext != '':
            new_block.append({
                "object": "block",
                "paragraph": {
                    "rich_text": [{
                        "text": {
                            "content": copy.deepcopy(linetext),
                        },
                    }],
                }
            })
        print(len(new_block))
        for segment in segment_list:
            subtitle = segment['text']
            newword = subtitle.split(']')[-1][1:]
            tempall = alltext + newword
            if len(tempall) > 2000:
                new_block.append({
                    "object": "block",
                    "paragraph": {
                        "rich_text": [{
                            "text": {
                                "content": copy.deepcopy(alltext),
                            },
                        }],
                    }
                })
                alltext = newword
            else:
                alltext += newword

        if alltext != '':
            new_block.append({
                "object": "block",
                "paragraph": {
                    "rich_text": [{
                        "text": {
                            "content": copy.deepcopy(alltext),
                        },
                    }],
                }
            })
            print(len(new_block))
        if len(new_block) <= 100:
            new_page = {
                "Name": {
                    "title": [{
                        "text": {
                            "content": self.app_options.note_name
                        }
                    }]
                },
            }

            self.notion.pages.create(parent={"database_id": database_id},
                                     properties=new_page,
                                     children=new_block)
        else:
            pagenum = len(new_block) // 100
            if len(new_block) % 100 != 0:
                pagenum += 1
            for j in range(1, pagenum):
                new_page = {
                    "Name": {
                        "title": [{
                            "text": {
                                "content":
                                self.app_options.note_name + '-' + str(j)
                            }
                        }]
                    },
                }
                self.notion.pages.create(parent={"database_id": database_id},
                                         properties=new_page,
                                         children=new_block[(j - 1) * 100:j *
                                                            100])
            new_page = {
                "Name": {
                    "title": [{
                        "text": {
                            "content":
                            self.app_options.note_name + '-' + str(pagenum)
                        }
                    }]
                },
            }
            self.notion.pages.create(parent={"database_id": database_id},
                                     properties=new_page,
                                     children=new_block[(pagenum - 1) * 100:])

    async def start_transcription(self):
        try:
            self.transcribing = True
            self.stream = create_audio_stream(self.app_options.audio_device,
                                              self.process_audio)
            self.stream.start()
            self._running.set()
            self._transcribe_task = asyncio.run_coroutine_threadsafe(
                self.transcribe_audio(), self.event_loop)
            eel.on_recive_message("Transcription started.")
            while self._running.is_set():
                await asyncio.sleep(1)
        except Exception as e:
            eel.on_recive_message(str(e))

    async def stop_transcription(self):
        try:
            self.transcribing = False
            if self._transcribe_task is not None:
                self.event_loop.call_soon_threadsafe(
                    self._transcribe_task.cancel)
                self._transcribe_task = None

            if self.app_options.create_audio_file and len(
                    self.all_audio_data_list) > 0:
                audio_data = np.concatenate(self.all_audio_data_list)

                path = "web/record"
                if not os.path.exists(path):
                    os.makedirs(path)
                if self.app_options.course != '':
                    path = "web/record/" + self.app_options.course
                    if not os.path.exists(path):
                        os.makedirs(path)

                self.all_audio_data_list.clear()
                write_audio(path, self.app_options.note_name, audio_data)
                self.batch_transcribe_audio(audio_data)

            if self.stream is not None:
                self._running.clear()
                self.stream.stop()
                self.stream.close()
                self.stream = None
                eel.on_recive_message("Transcription stopped.")
            else:
                eel.on_recive_message("No active stream to stop.")
        except Exception as e:
            eel.on_recive_message(str(e))
