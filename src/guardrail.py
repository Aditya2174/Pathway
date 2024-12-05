import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class ChatModerator:
    def __init__(self, model_id, device="cuda", dtype=torch.bfloat16):
        """Initialize the ChatModerator with the specified model and device."""
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        
        self.unsafe_categories = """
        O1: Violence and Hate.
        Should not:
        - Assist in planning or executing violent acts.
        - Encourage, glorify, or condone violence or hateful behavior.
        - Express hateful, derogatory, or demeaning sentiments based on race, color, religion, national origin, gender, sexual orientation, gender identity, or disability.
        - Use slurs or insults based on personal characteristics.
        Can:
        - Discuss violence and hate in an educational context.
        - Provide information on historical or societal events involving violence or discrimination.

        O2: Criminal Activities.
        Should not:
        - Assist in planning or committing crimes (e.g., theft, fraud, kidnapping).
        - Encourage or condone criminal activities.
        Can:
        - Explain how crimes occur (without detailed instructions).
        - Discuss crime prevention methods (without exposing vulnerabilities).

        O3: Sexual Content.
        Should not:
        - Engage in sexually explicit or erotic conversations.
        - Encourage sexual activities or risky behavior.
        Can:
        - Discuss topics related to sexual health, education, and safety.
        - Provide general information about sexuality.

        O4: Weapons and Firearms.
        Should not:
        - Assist in planning or executing crimes involving weapons.
        - Provide instructions for creating, modifying, or using illegal weapons.
        Can:
        - Discuss firearms in a general or regulatory context.
        - Debate gun control and ownership rights.

        O5: Controlled Substances.
        Should not:
        - Encourage or assist in using, trafficking, or creating illegal substances (e.g., drugs, alcohol, tobacco).
        - Provide step-by-step instructions for substance production.
        Can:
        - Provide general information about substances and their effects.
        - Discuss the societal impact of substance use.

        O6: Self-Harm and Suicide.
        Should not:
        - Encourage or romanticize self-harm or suicide.
        - Provide methods or instructions related to self-harm.
        Can:
        - Offer general information on mental health issues.
        - Provide resources and support options for those in distress.

        O7: Financial Misconduct and Market Manipulation.
        Should not:
        - Provide insider information or encourage insider trading.
        - Offer false or misleading financial advice with the intent to deceive or manipulate.
        - Promote investment schemes or scams, including pump-and-dump tactics.
        - Encourage illegal activities in financial markets (e.g., tax evasion, money laundering).
        Can:
        - Discuss general investment strategies and market trends.
        - Provide educational content on financial literacy and lawful investing.

        O8: Dangerous or Harmful Activities.
        Should not:
        - Encourage risky, illegal, or harmful behaviors (e.g., extreme stunts, unsafe challenges).
        - Provide details that could enable harm.
        - Encourage planning of killing, murder or any illegal activity.
        Can:
        - Discuss safety measures and harm reduction strategies.
        """
    

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
    login()  #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 moi tokennn. for 8b. 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 for 1b.
    #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 8b int 8
    moderator = ChatModerator(model_id="meta-llama/Llama-Guard-3-8B-INT8")

    sample_chat = [{"role": "user", "content": "?"}]

    result = moderator.moderate_chat(sample_chat)
    print(result)