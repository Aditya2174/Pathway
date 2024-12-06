import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from prompts import unsafe_categories

class ChatModerator:
    def __init__(self, model_id, device="cuda", dtype=torch.bfloat16):
        """Initialize the ChatModerator with the specified model and device."""
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
        
        self.unsafe_categories = unsafe_categories
    

    def generate_prompt(self, chat):
        """Generate the moderation prompt based on the chat content."""
        conversation = [turn["content"] for turn in chat]
        is_agent = len(conversation) % 2 == 0
        role = "Agent" if is_agent else "User"
        
        prompt = f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according to our safety policy with the below categories.

        <BEGIN UNSAFE CONTENT CATEGORIES>
        {self.unsafe_categories}
        <END UNSAFE CONTENT CATEGORIES>

        <BEGIN CONVERSATION>
        """
        for i, message in enumerate(conversation):
            role = "User" if i % 2 == 0 else "Agent"
            prompt += f"{role}: {message}\n\n"
        
        prompt += f"""<END CONVERSATION>

        Provide your safety assessment for {role} in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories.[/INST]"""
        
        return prompt

    def moderate_chat(self, chat):
        """Moderate the given chat using the model and return the assessment."""
        try:
            # Generate the moderation prompt
            prompt = self.generate_prompt(chat)
            
            # Tokenize and generate the output
            inputs = self.tokenizer([prompt], return_tensors='pt').to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.pad_token_id)
            
            # Decode the result and return the moderation assessment
            prompt_len = inputs["input_ids"].shape[-1]
            moderation_result = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
            return moderation_result
        
        except Exception as e:
            return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    login()
    moderator = ChatModerator(model_id="meta-llama/Llama-Guard-3-8B-INT8")

    sample_chat = [{"role": "user", "content": "?"}]

    result = moderator.moderate_chat(sample_chat)
    print(result)