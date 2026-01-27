# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The file has been adapted from hiyouga LLaMA-Factory project
# Copyright (c) 2025 LLaMA-Factory
# Licensed under the Apache License - https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE


import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from ..hparams import get_server_args
from ..utils.constants import EngineName

# import openai


# from ..extras.misc import torch_gc

if TYPE_CHECKING:
    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""General class for chat models. Backed by huggingface or vllm engines.

    Supports both sync and async methods.
    Sync methods: chat(), stream_chat() and get_scores().
    Async methods: achat(), astream_chat() and aget_scores().
    """

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        model_args, data_args, generating_args, finetuning_args, server_args = get_server_args(args)

        if model_args.infer_backend == EngineName.HF:
            from .hf_engine import HuggingfaceEngine

            self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        # elif model_args.infer_backend == EngineName.VLLM:
        #     try:
        #         from .vllm_engine import VllmEngine

        #         self.engine: BaseEngine = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        #     except ImportError as e:
        #         raise ImportError(
        #             "vLLM not install, you may need to run `pip install vllm`\n"
        #             "or try to use HuggingFace backend: --infer_backend huggingface"
        #         ) from e
        # elif model_args.infer_backend == EngineName.SGLANG:
        #     try:
        #         from .sglang_engine import SGLangEngine

        #         self.engine: BaseEngine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)
        #     except ImportError as e:
        #         raise ImportError(
        #             "SGLang not install, you may need to run `pip install sglang[all]`\n"
        #             "or try to use HuggingFace backend: --infer_backend huggingface"
        #         ) from e
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images=None,
        videos=None,
        audios=None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Get a list of responses of the chat model."""
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, videos, audios, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images=None,
        videos=None,
        audios=None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Asynchronously get a list of responses of the chat model."""
        return await self.engine.chat(messages, system, tools, images, videos, audios, **input_kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images=None,
        videos=None,
        audios=None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        r"""Get the response token-by-token of the chat model."""
        generator = self.astream_chat(messages, system, tools, images, videos, audios, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images=None,
        videos=None,
        audios=None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""Asynchronously get the response token-by-token of the chat model."""
        async for new_token in self.engine.stream_chat(
            messages, system, tools, images, videos, audios, **input_kwargs
        ):
            yield new_token

    def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Get a list of scores of the reward model."""
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Asynchronously get a list of scores of the reward model."""
        return await self.engine.get_scores(batch_input, **input_kwargs)


def run_chat() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            # torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})


# def run_chat(args: Optional[dict[str, Any]] = None) -> None:
#     """
#     Launch a conversation in the command line.
#     Note: Service-oriented deployment needs to be carried out using 'erniekit server' first.
#     """
#     args = read_args(args)
#     model_args, generating_args, finetuning_args, server_args = get_server_args(args)

#     messages = []
#     print(
#         "Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application."
#     )
#     print("Note: the command-line dialogue for VL-model only supports pure text input.")

#     ip = "0.0.0.0"
#     service_http_port = str(server_args.port)
#     client = openai.Client(
#         base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY"
#     )

#     while True:
#         try:
#             query = input("\nUser: ")
#         except UnicodeDecodeError:
#             print(
#                 "Detected decoding error at the inputs, please set the terminal encoding to utf-8."
#             )
#             continue
#         except Exception:
#             raise

#         if query.strip() == "exit":
#             break

#         if query.strip() == "clear":
#             messages = []
#             print("History has been removed.")
#             continue

#         messages.append({"role": "user", "content": query})
#         print("Assistant: ", end="", flush=True)

#         response = client.chat.completions.create(
#             model="default",
#             messages=messages,
#             temperature=generating_args.temperature,
#             top_p=generating_args.top_p,
#             max_tokens=generating_args.max_new_tokens,
#             frequency_penalty=generating_args.frequency_penalty,
#             presence_penalty=generating_args.presence_penalty,
#             stream=generating_args.stream,
#             stream_options=generating_args.stream_options,
#             extra_body={"enable_thinking": generating_args.enable_thinking},
#         )

#         assistant_response = ""

#         if generating_args.enable_thinking:
#             print_thinking = False
#             print_answer = False
#             if generating_args.stream:
#                 for chunk in response:
#                     if chunk.choices[0].delta.reasoning_content != "":
#                         if not print_thinking:
#                             print("thinking process:")
#                             print_thinking = True
#                         print(chunk.choices[0].delta.reasoning_content, end="")
#                         assistant_response += chunk.choices[0].delta.reasoning_content
#                     if chunk.choices[0].delta.content != "":
#                         if not print_answer:
#                             print("answer:")
#                             print_answer = True
#                         print(chunk.choices[0].delta.content, end="")
#                         assistant_response += chunk.choices[0].delta.content
#             else:
#                 print("thinking process:")
#                 print(response.choices[0].message.reasoning_content)
#                 print("answer:")
#                 print(response.choices[0].message.content)
#                 assistant_response += response.choices[0].message.reasoning_content
#                 assistant_response += response.choices[0].message.content
#         else:
#             if generating_args.stream:
#                 for chunk in response:
#                     if chunk.choices[0].delta is not None:
#                         print(chunk.choices[0].delta.content, end="")
#                         assistant_response += chunk.choices[0].delta.content
#             else:
#                 print(response.choices[0].message.content)
#                 assistant_response += response.choices[0].message.content
#         print()
#         messages.append({"role": "assistant", "content": assistant_response})
