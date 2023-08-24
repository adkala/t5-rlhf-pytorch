from typing import Any
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import pickle
import os


class TreeCreator:
    _prompt = PromptTemplate(
        input_variables=["product_name"],
        template="What is a one sentence description that accurately and thoroughly describes the product '{product_name}'?",
    )

    def __init__(self, openai_api_key=None, generated_descriptions_path=None):
        if openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0
            )
            self.chain = LLMChain(llm=self.llm, prompt=self._prompt)
        else:
            self.llm = None

        self.generated_descriptions = {}  # product: description
        if generated_descriptions_path:
            self.loadGeneratedDescriptions(generated_descriptions_path)


    def sampleTreeToCategoryTreeAndProducts(self, sampleTree, force_empty_descriptions=False):
        products = []
        product_categories = []

        def helper(sampleTree, cur=""):
            ct = []
            if isinstance(sampleTree, set):
                for item in sampleTree:
                    if self.llm and not force_empty_descriptions:
                        product_description = self.chain.run(item)
                        self.generated_descriptions[item] = product_description
                    else:
                        product_description = ""
                    products.append(
                        {
                            "product_name": item,
                            "product_description": product_description,
                        }
                    )
                    product_categories.append(
                        {"product_name": item, "category": cur[:-1]}
                    )
            else:
                for category in sampleTree:
                    t = {
                        "category_name": category,
                        "subcategories": helper(
                            sampleTree[category], cur + "%s/" % category
                        ),
                    }
                    ct.append(t)
            return ct

        category_tree = helper(sampleTree)
        return (category_tree, products), product_categories

    def saveGeneratedDescriptions(self, statePath="generated_descriptions.bin"):
        tmp = {}
        if os.path.isfile(statePath):
            with open(statePath, "rb") as f:
                tmp = pickle.load(statePath)
        with open(statePath, "wb") as f:
            pickle.dump(self.generated_descriptions | tmp, f)

    def loadGeneratedDescriptions(self, statePath="generated_descriptions.bin"):
        with open(statePath, "rb") as f:
            self._loadGeneratedDescriptions(pickle.load(statePath))

    def _loadGeneratedDescriptions(self, generated_descriptions):
        self.generated_descriptions = (
            generated_descriptions | self.generated_descriptions
        )
