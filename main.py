from opensourceai import OpenAI
from opensourceai import num_tokens_from_messages
import base64
import io


def toBase(path):
    return base64.b64encode(open(path,'rb').read()).decode()


client = OpenAI(api_key='key',url="http://127.0.0.1:7860/")





'''
from openai import OpenAI
client = OpenAI(api_key='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')


response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)

print(response)
'''

response = client.images.generate(
  prompt='A modern and luxurious interior space with a minimalist design, featuring a curvilinear, built-in swimming pool that mimics the shape of a river. The pool is at the forefront, with a sleek, reflective surface. Surrounding the pool, the floor has a glossy, polished finish, complementing the soft pink hue that dominates the room. Large, floor-to-ceiling windows provide a panoramic view, with sheer pink curtains diffusing the natural light that bathes the space. The furniture is contemporary and sparse, with a couple of low, round seats and a long, white lounge sofa set against the wall. The wall itself is curvaceous, adding to the fluidity of the space. The overall ambiance is serene, with a futuristic touch, bathed in a soft, monochromatic pink light that creates a dreamy and ethereal atmosphere. The time of day appears to be dusk, as the light coming through the windows suggests the sun is setting, casting a warm glow. The setting is devoid of people, emphasizing the tranquility and pristine condition of the space.',
  model="dall-e-3",
  size="1024x1024",
  response_format='url',
  quality="hd",
  #n=2
)
print(response)
url=response.data[0].url
print(url)



'''
out=client.images.edit(
  image=[toBase('test/img.jpg')],
  mask=toBase('test/mask.jpg'),
  prompt='An inflatable swimming ring in the shape of a flamingo.',
  model="dall-e-2",
  size="1024x1024",
  response_format='url'
)
print(out)
'''

'''
response =client.images.create_variation(
  image=toBase('images/1705148779.6832283.png'),
  model="dall-e-2",
  size="1024x1024",
  response_format='url',
  n=4
)
print(response)
url=response.data[0].url
print(url)
'''