import torch
from sentence_transformers import util, SentenceTransformer
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def retrieve_relevant_resources(
    query: str,
    embeddings: torch.tensor,
    embedding_model: SentenceTransformer,
    n_resources_to_return: int = 5,
):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices


def prompt_formatter(tokenizer,query: str, context_items: list[dict]) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [{"role": "user", "content": base_prompt}]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template, tokenize=False, add_generation_prompt=True
    )
    return prompt

import textwrap


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text


def load_llm_model(model_name):
    model_id = model_name
    use_quantization_config = True

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    attn_implementation = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype=torch.float16,  # datatype to use, we want float16
        quantization_config=quantization_config if use_quantization_config else None,
        low_cpu_mem_usage=False,  # use full memory
        attn_implementation=attn_implementation,
    )

    return llm_model, tokenizer


def ask(
    query,
    embeddings,
    llm_model_id,
    embedding_model_id,
    pages_and_chunks,
    n_resources_to_return,
    temperature=0.7,
    max_new_tokens=512,
    format_answer_text=True,
    return_answer_only=True,
):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
    """
    device = 'cuda'

    llm_model, tokenizer = load_llm_model(llm_model_id)

    embedding_model = SentenceTransformer(
        model_name_or_path=embedding_model_id, device=device
    )

    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        embedding_model=embedding_model,
        n_resources_to_return=n_resources_to_return,
    )

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()  # return score back to CPU

    # Format the prompt with context items
    prompt = prompt_formatter(tokenizer=tokenizer,query=query, context_items=context_items)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")


    # Generate an output of tokens
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,

    )

    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Replace special tokens and unnecessary help message
        output_text = (
            output_text.replace(prompt, "")
            .replace("<bos>", "")
            .replace("<eos>", "")
            .replace("Sure, here is the answer to the user query:\n\n", "")
        )

    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    keys_to_include = {"page_number", "sentence_chunk", "score"}
    context_items = [
        {key: item[key] for key in item if key in keys_to_include}
        for item in context_items
    ]

    return print_wrapped(output_text), context_items
