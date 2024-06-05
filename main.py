import argparse
import re
import time
import nemo.collections.asr as nemo_asr

from src.utils import extract_soap_sections
from llama_cpp import Llama
from pydub import AudioSegment
from transformers import pipeline


def pipeline_processing(path_to_wav: str, llm: bool = True, device: str = "cpu"):
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En", map_location=device)

    lcpp_llm = Llama(
        model_path="llama-2-7b-chat.Q4_K_M.gguf",
        n_ctx=4096,
        # n_threads=24,
    )  # CPU cores

    pipe = pipeline(
        "text-generation", model="omi-health/sum-small", trust_remote_code=True, max_length=2100, device=device
    )
    promt = (
        "You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response only follows the structured format: S:, O:, A:, P: without using markdown or special formatting and without your text. Avoid informal or conversational language in the all sections.",
    )

    # Validate wav, resample, converting
    audio = AudioSegment.from_file(path_to_wav)
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    audio.export("wavs/conv_med.wav", format="wav")

    # ASR
    text_from_asr = quartznet.transcribe(["wavs/conv_med.wav"])[0]
    soap_request = [
        {
            "role": "system",
            "content": promt,
        },
        {
            "role": "user", 
            "content": text_from_asr
        },
    ]
    if llm:
        outputs_from_model = lcpp_llm.create_chat_completion(soap_request)["choices"][0]["message"]["content"]
        results_soap = extract_soap_sections(outputs_from_model)
    else:
        outputs_from_model = pipe(soap_request)[0]["generated_text"]
        results_soap = extract_soap_sections(outputs_from_model)

    return text_from_asr, promt, results_soap, outputs_from_model


def main():
    parser = argparse.ArgumentParser(description="Process audio file and generate SOAP notes.")
    parser.add_argument("--wav_path", default="wavs/example_file.mp3", type=str, help="Path to the input wav file")
    parser.add_argument("--llm", default=True, type=bool, help="Use Llama model for text generation")
    parser.add_argument("--device", default="cpu", type=str, help="device for inference")

    args = parser.parse_args()

    text_from_asr, promt, results_soap, outputs_from_model = pipeline_processing(path_to_wav=args.wav_path, 
                                                                                 llm=args.llm, 
                                                                                 device=args.device)
    print('_' * 15)
    print('#' * 15)
    print(f"Results SOAP is \n {results_soap}")
    print('#' * 15)


if __name__ == "__main__":
    main()
