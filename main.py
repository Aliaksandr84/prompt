from langchain_openai import AzureChatOpenAI
from pprint import pprint

deployment_name = "gpt-4o-2024-05-13" # select anyone from the list above
temperature = 0.1

azure_llm = AzureChatOpenAI(
		deployment_name=deployment_name,
        temperature=temperature,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
	)

response = azure_llm.invoke("this is a try to test the model")
content = response.content
response_metadata = response.response_metadata
pprint(content)
pprint(response_metadata)

def perform_prompt(input_prompt: str):
    output_response = azure_llm.invoke(input_prompt)
    output_content = output_response.content
    pprint(output_content)