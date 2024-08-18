from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

class ChatbotQA:
    def __init__(self, retriever, model_name="llama2:7b-chat"):
        self.retriever = retriever
        self.model_name = model_name
        self.model = ChatOpenAI(model="gpt-4o-mini")
        self.memory = ""  # Initialize an empty memory string
        self.base_prompt_template = """Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"""
        self.prompt_template = self.base_prompt_template

    def setup_chain(self):
        """Set up the processing chain with the current prompt template."""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.chain = (
            RunnableParallel({"context": self.retriever, "question": RunnablePassthrough()})
            | prompt
            | self.model
            | StrOutputParser()
        )

    def update_prompt_template(self):
        """Update the prompt template to include the current memory."""
        self.prompt_template = f"{self.memory}\n{self.base_prompt_template}"

    def answer_question(self, question, st_callback):
        """Answer a given question using the processing chain."""
        # Define a simple class to encapsulate the question
        class Question(BaseModel):
            __root__: str

        # Update the memory and prompt template before answering
        self.memory += f"\nQuestion: {question}\n"
        self.update_prompt_template()
        self.setup_chain()  # Re-setup the chain with the updated prompt template

        self.chain = self.chain.with_types(input_type=Question)
        answer = self.chain.invoke(question)
        # Update memory with the answer
        self.memory += f"Answer: {answer}\n"

        return answer

if __name__ == "__main__":
    # Placeholder for the retriever object, replace with actual retriever from the embedding generation step
    retriever = None

    chatbot = ChatbotQA(retriever)
    chatbot.setup_chain()

    while True:
        user_input = input("Human: ")
        if user_input.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        else:
            response = chatbot.answer_question(user_input)
            print("AI Assistant:", response)
