import requests as rq
from PIL import Image as PilImage
import tiktoken
import base64
import time
import re
import io

class OpenAIError(Exception):
    pass



class OpenAI:
    def __init__(self, api_key, url):
        if not api_key:
            raise OpenAIError("The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable")
        if not url:
            raise OpenAIError("The url client option must be set either by passing url to the client or by setting the OPENAI_API_URL environment variable")
        self.api_key = api_key
        self.url = url
        self.images = self.Images(url)
        self.chat = self.Chat(url)
    

    class Audio:
        class Speech:
            # https://platform.openai.com/docs/api-reference/audio/createSpeech
            # Create speech
            def create(model='tts-1',input=None,voice=None,response_format='mp3',speed=1):
                # [Required] model: One of the available TTS models: `tts-1` or `tts-1-hd`
                # [Required] input: The text to generate audio for. The maximum length is 4096 characters.
                # [Required] voice: The voice to use when generating the audio. Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`. Previews of the voices are available in the Text to speech guide.
                # (Optional) response_format: The format to audio in. Supported formats are `mp3`, o`pus`, `aac`, and `flac`.
                # (Optional) speed: The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.
                return ''
        class Transcriptions:
            def create(file=None,model=None,language=None,prompt=None,response_format='json',temperature=0):
                # [Required] file: The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
                # [Required] model: ID of the model to use. Only `whisper-1` is currently available.
                # (Optional) language: The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
                # (Optional) prompt: An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
                # (Optional) response_format: The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
                # (Optional) temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
                return ''
        class Translations:
            def create(file=None,model=None,language=None,prompt=None,response_format='json',temperature=0):
                # [Required] file: The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
                # [Required] model: ID of the model to use. Only `whisper-1` is currently available.
                # (Optional) language: The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
                # (Optional) prompt: An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
                # (Optional) response_format: The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
                # (Optional) temperature: The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
                return ''
    class Chat:
        class Completions:
            def create(self, model, messages, frequency_penalty=0, logit_bias=None, logprobs=False, top_logprobs=None, max_tokens=None, n=1, presence_penalty=0, response_format=None, seed=None, stop=None, stream=False, temperature= 1,top_p=1,tools=[],user=None,prompt=None):
                # prompt or messages
                # [Required] messages
                # [Required] model
                # (Optional) frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
                # (Optional) logit_bias: Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
                # (Optional) logprobs: Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`. This option is currently not available on the `gpt-4-vision-preview` model.
                # (Optional) top_logprobs: An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.
                # (Optional) max_tokens
                # (Optional) n
                # (Optional) presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
                # (Optional) response_format: An object specifying the format that the model must output. Compatible with `gpt-4-1106-preview` and `gpt-3.5-turbo-1106`.\nSetting to `{ "type": "json_object" }` enables JSON mode, which guarantees the message the model generates is valid JSON.\nImportant: when using JSON mode, you must also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if `finish_reason="length"`, which indicates the generation exceeded `max_tokens` or the conversation exceeded the max context length.
                # (Optional) response_format = { "type": "json_object" }
                # "content": "{\"winner\": \"Los Angeles Dodgers\"}"`
                # (Optional) Seed: This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.
                # (Optional) stop: Up to 4 sequences where the API will stop generating further tokens.
                # (Optional) stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Example Python code[link].
                # (Optional) temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.\nWe generally recommend altering this or top_p but not both.
                # (Optional) top_p
                # (Optional) tools [array] : A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.
                # (Optional) tool_choice: Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via {"type": "function", "function": {"name": "my_function"}} forces the model to call that function.\nnone is the default when no functions are present. auto is the default if functions are present.
                # (Optional) user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.
                # {Deprecated} function_call
                # {Deprecated} functions
                message = ChatCompletionMessage(content='hello', role='assistant')
                choice = Choice(finish_reason='stop', index=0, message=message)
                usage = CompletionUsage(completion_tokens=24, prompt_tokens=31, total_tokens=55)
                return ChatCompletion(id='chatcmpl-8gYgXHDtv5U7usc8hWlInJvF7sDjY', choices=[choice], created=int(time.time()), model=model, object='chat.completion', system_fingerprint='fp_cbe4fa03fe', usage=usage)
        
        # Function calling - Deprecated
        
        def __init__(self, url):
            self.url = url
            self.completions = self.Completions()
        
        
    class Embeddings:
        def create(self,input=None,model=None,encoding_format='float',user=None):
            # [Required] input: Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for text-embedding-ada-002), cannot be an empty string, and any array must be 2048 dimensions or less. Example Python code for counting tokens.
            #  [Required] model: ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
            # (Optional) encoding_format: The format to return the embeddings in. Can be either float or base64.
            # (Optional) user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more
            return ''
    
    class Fine_tuning:
        class Jobs:
            def create():
                return 'function:fineturning'
            def list():
                return 'procedure:fineturning'
            def list_events():
                return 'function:fineturning'
            def retrieve():
                return 'function:fineturning'
            def cancel():
                return 'function:fineturning'
    
    class Files:
        def create():
            return 'function:fineturning'
        def list():
            return 'procedure:fineturning'
        def retrieve():
            return 'function:fineturning'
        def delete():
            return 'function:fineturning'
        def retrieve_content():
            return 'function:fineturning'
    
    class Images:
        def __init__(self, url):
            self.url = url
        
        # https://api.openai.com/v1/images/generations
        # Create image
        def generate(self, prompt, model='dall-e-2', response_format='url', n=1, size="1024x1024", quality='standard', style=None, timeout=5*60,user=None):
            # response_format: url or b64_json
            # style: vivid or natural , only dall-e-3
            # user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
            
            return converPromtSDXL(prompt, model, n, size, quality, self.url, timeout, response_format)
        
        # https://api.openai.com/v1/images/edits 
        # Create image edit (only dall-e-2)
        def edit(self,image,prompt,mask,model='dall-e-2',n=1,size='1024x1024',response_format='url', timeout=5*60,user=None):
            # user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
            
            return convertPromptSDXLInpaint(mask, image, prompt, model, n, size, self.url, timeout, response_format)
            
        
        # https://platform.openai.com/docs/api-reference/images/createVariation
        # Create image variation
        
        def create_variation(self, image, model='dall-e-2', n=1, response_format='url', size='1024x1024', timeout=5*60,user=None):
            # user: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse
            # The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
            # The number of images to generate. Must be between 1 and 10. For dall-e-3, only n=1 is supported.
            return convertPromptSDXLImg2Img(image, model, n, size, self.url, timeout, response_format)
            
    
    class Models:
        def list():
            return 'procedure:fineturning'
        def retrieve():
            return 'function:fineturning'
        def delete():
            return 'function:fineturning'
    
    class Moderations:
        def create(self,input=None,model=None):
            return ''
            # [Required] input: The input text to classify
            # (Optional) model: Two content moderations models are available: text-moderation-stable and text-moderation-latest.\n\nThe default is text-moderation-latest which will be automatically upgraded over time. This ensures you are always using our most accurate model. If you use text-moderation-stable, we will provide advanced notice before updating the model. Accuracy of text-moderation-stable may be slightly lower than for text-moderation-latest.

#----------------------------#
#           Class            #
#----------------------------#

# Image #

class ImagesResponse:
    def __init__(self, created, data):
        self.created = created
        self.data = data

    def __repr__(self):
        return f"ImagesResponse(created={self.created}, data=[{', '.join(map(str, self.data))}])"

    def __getitem__(self, key):
        return getattr(self, key, None)

class Image:
    def __init__(self, b64_json=None, revised_prompt=None, url=None):
        self.b64_json = b64_json
        self.revised_prompt = revised_prompt
        self.url = url

    def __repr__(self):
        return f"Image(b64_json={self.b64_json}, revised_prompt={self.revised_prompt}, url='{self.url}')"


# Text generation #

class Chat:
    def __init__(self, messages=None):
        self.messages = messages or []

    def __repr__(self):
        return "\n".join(map(str, self.messages))

    @staticmethod
    def parse(chat_str):
        messages = []
        for line in chat_str.split("\n"):
            if line.startswith("Message("):
                parts = line[len("Message("):-1].split(", ")
                role = parts[0].split(": .")[1]
                text = parts[1].split(": \"")[1][:-1]
                messages.append(Message(role, text))
        return Chat(messages)

class Message:
    def __init__(self, role, text):
        self.role = role
        self.text = text

    def __repr__(self):
        return f"Message(role: .{self.role}, text: \"{self.text}\")"


# Util #



class CompletionUsage:
    def __init__(self, completion_tokens, prompt_tokens, total_tokens):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.total_tokens = total_tokens

    def __repr__(self):
        return f'CompletionUsage(completion_tokens={self.completion_tokens}, prompt_tokens={self.prompt_tokens}, total_tokens={self.total_tokens})'

class ChatCompletionMessage:
    def __init__(self, content, role):
        self.content = content
        self.role = role

    def __repr__(self):
        return f'ChatCompletionMessage(content=\'{self.content}\', role=\'{self.role}\')'

class Choice:
    def __init__(self, finish_reason, index, message, logprobs=None):
        self.finish_reason = finish_reason
        self.index = index
        self.message = message
        self.logprobs = logprobs

    def __repr__(self):
        return f'Choice(finish_reason=\'{self.finish_reason}\', index={self.index}, message={self.message}, logprobs={self.logprobs})'

class ChatCompletion:
    def __init__(self, id, choices, created, model, object, system_fingerprint, usage):
        self.id = id
        self.choices = choices
        self.created = created
        self.model = model
        self.object = object
        self.system_fingerprint = system_fingerprint
        self.usage = usage

    def __repr__(self):
        choices_repr = '[' + ', '.join(map(str, self.choices)) + ']'
        return f'ChatCompletion(id=\'{self.id}\', choices={choices_repr}, created={self.created}, model=\'{self.model}\', object=\'{self.object}\', system_fingerprint=\'{self.system_fingerprint}\', usage={self.usage})'


#----------------------------#
#           Parse            #
#----------------------------#

def parse_images_response(response_str):
    created_pattern = r"ImagesResponse\(created=(\d+), data=\[(.*)\]\)"
    match = re.match(created_pattern, response_str)
    if match:
        created, data_str = match.groups()
        images = [parse_image(image_str.strip()) for image_str in data_str.split(", Image")]
        return ImagesResponse(created=int(created), data=images)
    return None

def parse_image(image_str):
    pattern = r"Image\(b64_json=(.*), revised_prompt=(.*), url='(.*)'\)"
    match = re.match(pattern, image_str)
    if match:
        b64_json, revised_prompt, url = match.groups()
        b64_json = None if b64_json == 'None' else b64_json
        revised_prompt = None if revised_prompt == 'None' else revised_prompt
        return Image(b64_json=b64_json, revised_prompt=revised_prompt, url=url)
    return None

def parse_chat(chat_str):
    messages = []
    for line in chat_str.split("\n"):
        if line.startswith("Massage("):
            parts = line[len("Massage("):-1].split(", text: ")
            role = parts[0].split(": .")[1]
            text = parts[1].strip("\"")
            messages.append(Message(role, text))
    return messages

#----------------------------#
#          Realise           #
#----------------------------#


def converPromtSDXL(prompt,model,n,size,quality,url,timeout,response_format):
    # Tested
    size_error  = {'error': {'code': None, 'message': f"'{size}' is not one of ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024'] - 'size'", 'param': None, 'type': 'invalid_request_error'}}
    quality_error = {'error': {'code': None, 'message': f"{quality} is not one of ['standard', 'hd'] - 'quality'", 'param': None, 'type': 'invalid_request_error'}}
    model_size_unsupport = {'error': {'code': 'invalid_size', 'message': 'The size is not supported by this model.', 'param': None, 'type': 'invalid_request_error'}}
    model_error = {'error': {'code': None, 'message': f'Invalid model {model}. The model argument should be left blank.', 'param': None, 'type': 'invalid_request_error'}}
    n_error={'error': {'code': None, 'message': f"{n} is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    response_format_error= {'error': {'code': None, 'message': "0 is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    
    # Untested - OpenAi
    
    
    # Code
    
    if response_format not in ['url','b64_json']:
        raise OpenAIError('Error code: 400 - ' + str(response_format_error))
    
    if n<1 or n>10:
        raise OpenAIError('Error code: 400 - ' + str(n_error))
    if quality not in ['standard','hd']:
        raise OpenAIError('Error code: 400 - ' + str(quality_error))
    
    
    if size in ['256x256','512x512','1024x1024','1792x1024','1024x1792']:
        if model=='dall-e-2':
            if size=='256x256':
                width,height=256,256
            elif size=='512x512':
                width,height=512,512
            elif size=='1024x1024':
                width,height=1024,1024
            else:
                raise OpenAIError('Error code: 400 - ' + str(model_size_unsupport))
        elif model=='dall-e-3':
            if size=='1024x1024':
                width,height=1024,1024
            elif size=='1792x1024':
                width,height=1792,1024
            elif size=='1024x1792':
                width,height=1024,1792
            else:
                raise OpenAIError('Error code: 400 - ' + str(model_size_unsupport))
        else:
            raise OpenAIError('Error code: 400 - ' + str(model_error))
    else:
        raise OpenAIError('Error code: 400 - ' + str(size_error))
    api='sdapi/v1/txt2img'
    
    template = {
      "prompt": prompt,
      "negative_prompt": "",
      "seed": -1,
      "sampler_name": "Euler a",
      "steps": {'standard':20,'hd':30}[quality],
      "cfg_scale": {'standard':5,'hd':7}[quality],
      "width": width,
      "height": height,
      "restore_faces": False,
      "do_not_save_samples": False,
      "do_not_save_grid": False,
      "n_iter": n,
    }
    
    out=rq.post(url+api,json=template,timeout=timeout).json()
    images=out['images']
    data=[]
    for i in range(len(images)):
        name = time.time()
        
        
        if response_format=='url':
            url=f'images/{name}.png'
            file=open(f'images/{name}.png','wb')
            data.append(Image(b64_json=None, revised_prompt=prompt, url=url))
            file.write(base64.b64decode(images[i]))
            file.close()
        else:
            data.append(Image(b64_json=images[i], revised_prompt=prompt, url=None))
        
    return ImagesResponse(created=int(time.time()), data=data)

def convertPromptSDXLImg2Img(init_image, model, n, size, url, timeout, response_format):
    # Tested
    size_error  = {'error': {'code': None, 'message': f"'{size}' is not one of ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024'] - 'size'", 'param': None, 'type': 'invalid_request_error'}}
    model_size_unsupport = {'error': {'code': 'invalid_size', 'message': 'The size is not supported by this model.', 'param': None, 'type': 'invalid_request_error'}}
    model_error = {'error': {'code': None, 'message': f'Invalid model {model}. The model argument should be left blank.', 'param': None, 'type': 'invalid_request_error'}}
    n_error={'error': {'code': None, 'message': f"{n} is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    response_format_error= {'error': {'code': None, 'message': "0 is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    model_chose_err={'error':'err'}
    url_to_base64={'error':'err'}
    # Untested - OpenAi
    
    prompt=''
    # Code
    
    if response_format not in ['url','b64_json']:
        raise OpenAIError('Error code: 400 - ' + str(response_format_error))
    
    if n<1 or n>10:
        raise OpenAIError('Error code: 400 - ' + str(n_error))
    
    
    if size in ['256x256','512x512','1024x1024','1792x1024','1024x1792']:
        if model=='dall-e-2':
            if size=='256x256':
                width,height=256,256
            elif size=='512x512':
                width,height=512,512
            elif size=='1024x1024':
                width,height=1024,1024
            else:
                raise OpenAIError('Error code: 400 - ' + str(model_size_unsupport))
        elif model=='dall-e-3':
            raise OpenAIError('Error code: 400 - ' + str(model_chose_err))
        else:
            raise OpenAIError('Error code: 400 - ' + str(model_chose_err))
    else:
        raise OpenAIError('Error code: 400 - ' + str(size_error))
    api='sdapi/v1/img2img'
    
    template = {
      "init_images": [init_image],
      "prompt": prompt,
      "negative_prompt": "",
      "seed": -1,
      "sampler_name": "Euler a",
      "steps": 30,
      "cfg_scale": 7,
      "width": width,
      "height": height,
      "restore_faces": False,
      "do_not_save_samples": False,
      "do_not_save_grid": False,
      "n_iter": n,
      "denoising_strength": 0.75,
    }

    out=rq.post(url+api,json=template,timeout=timeout).json()
    
    
    try:
        images=out['images']
    except:
        raise OpenAIError('Error code: 400 - ' + str(url_to_base64))
        
    data=[]
    for i in range(len(images)):
        name = time.time()
        if response_format=='url':
            url=f'images/{name}.png'
            file=open(f'images/{name}.png','wb')
            data.append(Image(b64_json=None, revised_prompt=prompt, url=url))
            file.write(base64.b64decode(images[i]))
            file.close()
        else:
            data.append(Image(b64_json=images[i], revised_prompt=prompt, url=None))
        
    
    return ImagesResponse(created=int(time.time()), data=data)


def convertPromptSDXLInpaint(mask, init_image, prompt, model, n, size, url, timeout, response_format):
    # Tested
    size_error  = {'error': {'code': None, 'message': f"'{size}' is not one of ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024'] - 'size'", 'param': None, 'type': 'invalid_request_error'}}
    model_size_unsupport = {'error': {'code': 'invalid_size', 'message': 'The size is not supported by this model.', 'param': None, 'type': 'invalid_request_error'}}
    model_error = {'error': {'code': None, 'message': f'Invalid model {model}. The model argument should be left blank.', 'param': None, 'type': 'invalid_request_error'}}
    n_error={'error': {'code': None, 'message': f"{n} is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    response_format_error= {'error': {'code': None, 'message': "0 is less than the minimum of 1 - 'n'", 'param': None, 'type': 'invalid_request_error'}}
    model_chose_err={'error':'err'}
    # Untested - OpenAi
    
    # Code
    
    if response_format not in ['url','b64_json']:
        raise OpenAIError('Error code: 400 - ' + str(response_format_error))
    
    if n<1 or n>10:
        raise OpenAIError('Error code: 400 - ' + str(n_error))
    
    
    if size in ['256x256','512x512','1024x1024','1792x1024','1024x1792']:
        if model=='dall-e-2':
            if size=='256x256':
                width,height=256,256
            elif size=='512x512':
                width,height=512,512
            elif size=='1024x1024':
                width,height=1024,1024
            else:
                raise OpenAIError('Error code: 400 - ' + str(model_size_unsupport))
        elif model=='dall-e-3':
            raise OpenAIError('Error code: 400 - ' + str(model_chose_err))
        else:
            raise OpenAIError('Error code: 400 - ' + str(model_chose_err))
    else:
        raise OpenAIError('Error code: 400 - ' + str(size_error))
    api='sdapi/v1/img2img'
    
    
    for i in range(len(init_image)):
        init_image[i]=resize_image(init_image[i],width,height)
    mask=resize_image(mask,width,height)
    
    template = {
      "init_images": init_image,
      "mask": mask,
      "prompt": prompt,
      "negative_prompt": "",
      "seed": -1,
      "sampler_name": "Euler a",
      "steps": 30,
      "cfg_scale": 7,
      "width": width,
      "height": height,
      "restore_faces": False,
      "do_not_save_samples": False,
      "do_not_save_grid": False,
      "n_iter": n,
      "denoising_strength": 0.75,
      'inpainting_fill':1
    }

    out=rq.post(url+api,json=template,timeout=timeout).json()
    
    try:
        images=out['images']
    except:
        raise OpenAIError('Error code: 400 - ' + str(out))
    
    data=[]
    for i in range(len(images)):
        name = time.time()
        
        
        if response_format=='url':
            url=f'images/{name}.png'
            file=open(f'images/{name}.png','wb')
            data.append(Image(b64_json=None, revised_prompt=prompt, url=url))
            file.write(base64.b64decode(images[i]))
            file.close()
        else:
            data.append(Image(b64_json=images[i], revised_prompt=prompt, url=None))
        
    return ImagesResponse(created=int(time.time()), data=data)



def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
  """Returns the number of tokens used by a list of messages."""
  try:
      encoding = tiktoken.encoding_for_model(model)
  except KeyError:
      encoding = tiktoken.get_encoding("cl100k_base")
  if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
      num_tokens = 0
      for message in messages:
          num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":  # if there's a name, the role is omitted
                  num_tokens += -1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens
  else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def resize_image(img_base64, w, h):
    # Декодирование изображения из base64
    img_data = base64.b64decode(img_base64)
    img = PilImage.open(io.BytesIO(img_data))

    # Изменение размера изображения
    resized_img = img.resize((w, h))

    # Кодирование обратно в base64
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    img_base64_resized = base64.b64encode(buffered.getvalue()).decode()

    return img_base64_resized