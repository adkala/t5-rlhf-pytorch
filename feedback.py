from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

import json

BASE_PROMPT = r"""You are a program that is given a category tree name 'category_tree' and a list of products named 'products' represented in JSON format. It will be in the format given below. For clarity, there are example inputs which will be explained below.

category_tree = [{
   "category_name":"Electronics",
   "subcategories":[
      {
         "category_name":"Computers",
         "subcategories":[
            {
               "category_name":"Laptops",
               "subcategories":[

               ]
            },
            {
               "category_name":"Desktops",
               "subcategories":[

               ]
            }
         ]
      },
      {
         "category_name":"Home Entertainment",
         "subcategories":[
            {
               "category_name":"TVs",
               "subcategories":[

               ]
            },
            {
               "category_name":"Speakers",
               "subcategories":[

               ]
            }
         ]
      }
   ]
}]

products = [
   {
      "product_name":"Example Laptop",
      "product_description":"This is an example laptop.",
   },
   {
      "product_name":"Example TV",
      "product_description":"This is an example TV.",
   },
   {
      "product_name":"Example Speaker",
      "product_description":"This is an example speaker.",
   },
   {
      "product_name":"BMW X5",
      "product_description":"This is an example car.",
   }
]

The category tree is represented as a JSON object with the following properties:
- category_name: a string representing the name of the category.
- subcategories: a list of subcategories, where each subcategory is a category tree object.

Each product is represented as a JSON object with the following properties:
- name: a string representing the name of the product.
- description: a string representing the description of the product.

For each product in products, return a JSON object with the following properties according to the name and description of each product:
- name: a string representing the name of the product.
- category: a string representing the full path of the leaf node category that best matches the product attributes, where each level is separated by a forward slash (/). If no matching category is found, the value of category is 'null'.

For clarity, for the input of category_tree and products above, you should return the following and only the following in JSON format and as a raw string:

output = [
  {
    "product_name": "Example Laptop",
    "category": "Electronics/Computers/Laptops"
  },
  {
    "product_name": "Example TV",
    "category": "Electronics/Home Entertainment/TVs"
  },
  {
    "product_name": "Example Speaker",
    "category": "Electronics/Home Entertainment/Speakers"
  },
  {
    "product_name": "BMW X5",
    "category": null
  }
]"""

TEMPLATE = r"""What would you return for the following category_tree and products JSON objects?

Return nothing but the raw json object with no prepending text explaining the output. Your message should start and end with the square brackets.

category_tree = {category_tree}

products = {products}"""

class GPTFeedback:
  def __init__(self, openai_api_key, base_prompt=BASE_PROMPT, template=TEMPLATE, model='gpt-3.5-turbo'):
    self.base_prompt = base_prompt
    self.template = template

    self.llm = ChatOpenAI(openai_api_key=openai_api_key, model=model, temperature=0)
    self.prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=(
                self.base_prompt
            )
        ),
        HumanMessagePromptTemplate.from_template(self.template)
    ])

    self.memory = []

  def getCategories(self, category_tree: list, products: list):
    response = self.llm(self.prompt_template.format_messages(category_tree=json.dumps(category_tree), products=json.dumps(products)))
    try:
      content = json.loads(response.content)
    except:
      print("LLM returned unparsable object. Please try different inputs.")
      content = None

    self.memory.append((category_tree, products, content))
    return content

  def getMemory(self):
    return self.memory